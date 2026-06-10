"""The local-test-mode harness: ``TestMode`` (async context) + ``TestPipeline``.

``pipeline.test_mode(...)`` returns a ``TestMode``. Entering it builds the scoped
runtime (registry overlay, test engine, in-memory audit sink + evidence store) and
yields a ``TestPipeline`` whose ``run`` delegates to the REAL ``Pipeline.run`` with
those scoped objects via the ``registry_override`` seam. No global state is mutated,
so exit clears nothing. Test mode fakes externals; Drawbore decides success."""

from __future__ import annotations

from typing import Any, Collection, Mapping

from pydantic import BaseModel

from drawbore.audit import AuditSink, InMemoryAuditSink
from drawbore.evidence import EvidenceStore, InMemoryEvidenceStore
from drawbore.llm import CredentialChecker, LLMRuntimeConfig
from drawbore.state import CheckpointStore
from drawbore.identity import IdentityRegistry
from drawbore.tools import TrustLabel

from .engine import TestEngine
from .errors import TestingError
from .tools import build_scoped_registry


def _validate_mock_agent_keys(
    pipeline,
    *,
    model_responses: Mapping[str, Any],
    loop_scripts: Mapping[str, Any],
) -> None:
    """Eagerly reject a model/loop mock keyed by an agent the pipeline has no matching
    step for (a typo guard mirroring the tool-mock guard in ``build_scoped_registry``).
    ``mock_model_responses`` is consumed only for one-shot model agents (model
    set, no tools); ``mock_loop_scripts`` only for model+tools agents — so a key
    outside its category is a dead entry that would otherwise fail closed silently at
    run time. Surfacing it at ``test_mode(...)`` makes the mistake legible immediately."""
    from drawbore.pipeline.graph import JoinNode
    one_shot: set[str] = set()
    model_tools: set[str] = set()
    for step in pipeline.steps:
        if isinstance(step, JoinNode):
            continue
        spec = step.agent.spec
        if spec.model is None:
            continue
        if spec.tools:
            model_tools.add(spec.name)
        else:
            one_shot.add(spec.name)
    for name in model_responses:
        if name not in one_shot:
            raise TestingError(
                f"mock_model_responses names agent '{name}' but no pipeline step is a "
                f"one-shot model agent (model set, no tools) with that name"
            )
    for name in loop_scripts:
        if name not in model_tools:
            raise TestingError(
                f"mock_loop_scripts names agent '{name}' but no pipeline step is a "
                f"model+tools agent (model set, tools declared) with that name"
            )


class TestPipeline:
    """The active test wrapper. ``run`` delegates to the real ``Pipeline.run``."""

    def __init__(self, mode: "TestMode") -> None:
        self._mode = mode

    @property
    def audit_sink(self) -> AuditSink:
        return self._mode._audit_sink

    @property
    def evidence_store(self) -> EvidenceStore:
        return self._mode._evidence_store

    async def run(
        self,
        initial: BaseModel,
        *,
        run_id: str | None = None,
        checkpoints: CheckpointStore | None = None,
        identities: IdentityRegistry | None = None,
        tenant_id: str | None = None,
        audit: AuditSink | None = None,
        evidence_store: EvidenceStore | None = None,
        initial_trust: TrustLabel = TrustLabel.TRUSTED,
    ):
        mode = self._mode
        rid = run_id or mode._next_run_id()
        # A per-run evidence_store override (e.g. a deliberately failing store) is
        # honored for BOTH compression and retrieval: rebind the overlay's
        # evidence://retrieve to it for this run so they agree.
        store = evidence_store if evidence_store is not None else mode._evidence_store
        # Rebuild the overlay fresh for THIS run so sequence-form tool mocks reset each
        # run (matching loop-script semantics) and evidence://retrieve binds the
        # resolved store for both compression and retrieval.
        registry = build_scoped_registry(
            mode._pipeline, mock_tools=mode._mock_tools,
            allow_real_tools=mode._allow_real_tools, evidence_store=store,
        )
        return await mode._pipeline.run(
            initial,
            engine=mode._engine,
            run_id=rid,
            checkpoints=checkpoints,
            identities=identities,
            tenant_id=tenant_id,
            audit=audit if audit is not None else mode._audit_sink,
            evidence_store=store,
            registry_override=registry,
            initial_trust=initial_trust,
        )


class TestMode:
    """Async context manager owning the scoped test runtime."""

    def __init__(
        self,
        pipeline,
        *,
        mock_tools: Mapping[str, Any] | None = None,
        mock_model_responses: Mapping[str, Any] | None = None,
        mock_loop_scripts: Mapping[str, Any] | None = None,
        allow_real_tools: Collection[str] = (),
        evidence_store: EvidenceStore | None = None,
        audit_sink: AuditSink | None = None,
        run_id_prefix: str | None = None,
        max_llm_calls: int = 8,
        llm_config: LLMRuntimeConfig | None = None,
        credential_checker: CredentialChecker | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._mock_tools = dict(mock_tools or {})
        self._allow_real_tools = tuple(allow_real_tools)
        self._audit_sink = audit_sink if audit_sink is not None else InMemoryAuditSink()
        self._evidence_store = (
            evidence_store if evidence_store is not None else InMemoryEvidenceStore()
        )
        model_responses = dict(mock_model_responses or {})
        loop_scripts = dict(mock_loop_scripts or {})
        # Eager fail-closed validation of mock keys: a model/loop mock keyed by a name
        # outside its consumed category is a dead entry — surface it now, not at a later
        # run-time halt (symmetric with the tool-mock guard below).
        _validate_mock_agent_keys(
            pipeline, model_responses=model_responses, loop_scripts=loop_scripts,
        )
        self._engine = TestEngine(
            model_responses=model_responses,
            loop_scripts=loop_scripts,
            max_llm_calls=max_llm_calls,
            llm_config=llm_config,
            credential_checker=credential_checker,
        )
        # Eager fail-closed validation: building now surfaces an undeclared-tool mock
        # (or an allow_real_tools typo) at test_mode(...) construction rather than at
        # first run. The actual per-run overlay is rebuilt fresh in TestPipeline.run
        # so sequence-form tool mocks reset each run (matching loop-script semantics).
        build_scoped_registry(
            pipeline, mock_tools=self._mock_tools,
            allow_real_tools=self._allow_real_tools, evidence_store=self._evidence_store,
        )
        self._prefix = run_id_prefix or f"{pipeline.name}-test"
        self._run_counter = 0

    def _next_run_id(self) -> str:
        self._run_counter += 1
        return f"{self._prefix}-{self._run_counter}"

    async def __aenter__(self) -> TestPipeline:
        return TestPipeline(self)

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        # No global state was mutated; nothing to undo.
        return False
