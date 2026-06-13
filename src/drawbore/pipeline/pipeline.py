"""Pipeline composition and cross-step orchestration policy.

Drawbore owns cross-step policy: binding-based payload construction, the schema
gate at every edge, and halt-on-violation. The engine runs only individual steps.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Collection, Literal, Mapping

from pydantic import BaseModel

from drawbore.agent import Agent
from drawbore.audit import AuditRecord, AuditRecorder, AuditSink, RunMetrics
from drawbore.context import build_input, sanitize
from drawbore.evidence import (
    EVIDENCE_TOOL_REF,
    EvidencePolicy,
    EvidenceStore,
    register_evidence_tool,
)
from drawbore.observability import payload_hash
from drawbore.errors import SanitizationError
from drawbore.schema import check_compatibility, validate
from drawbore.schema.errors import SchemaCompatibilityError, SchemaValidationError
from drawbore.state import (
    CheckpointStore,
    EffectLedger,
    ResumeLedger,
    ResumeLedgerBuilder,
    RunState,
)
from drawbore.state.step_seal import diff_seals, field_label, seal_for
from drawbore.orchestration import LocalEngine, OrchestratorEngine
from drawbore.tools import (
    ToolAccessError,
    ToolProxy,
    ToolRegistry,
    TokenIssuer,
    registry as default_registry,
)
from drawbore.tools.taint import TaintLedger, TrustLabel, join
from drawbore.escalation import (
    ApprovalDecision,
    ApprovalRequest,
    EscalationDispatcher,
    EscalationPackage,
    EscalationPolicy,
    HasConfidence,
    RecordingDispatcher,
    build_escalation,
)
from drawbore.identity import IdentityRegistry

from .binding import From
from .conditions import When
from .executor import StepExecutor
from .outcome import Halt


@dataclass
class Step:
    agent: Agent
    inputs: dict[str, From]
    depends_on: list[str] = field(default_factory=list)
    evidence: EvidencePolicy | None = None
    when: "When | None" = None


@dataclass
class RunResult:
    status: Literal["completed", "halted", "escalated"]
    outputs: dict[str, BaseModel]
    steps_run: int
    halted_at: str | None = None
    reason: str | None = None
    escalations: list[EscalationPackage] = field(default_factory=list)
    audit_trace: "AuditRecord | None" = None
    run_id: str | None = None
    halt_code: str | None = None
    resume_ledger: "ResumeLedger | None" = None
    metrics: "RunMetrics | None" = None
    approval_request: "ApprovalRequest | None" = None


class Pipeline:
    """A topology of typed agents with explicit data bindings."""

    def __init__(
        self,
        name: str,
        version: str = "0.0.0",
        registry=None,
        *,
        on_failure: EscalationPolicy | None = None,
        dispatcher: EscalationDispatcher | None = None,
        confidence_threshold: float | None = None,
    ):
        self.name = name
        self.version = version
        self.steps: list["Step | JoinNode"] = []
        self._by_name: dict[str, "Step | JoinNode"] = {}
        self._registry = registry if registry is not None else default_registry
        self._on_failure = on_failure
        self._dispatcher = dispatcher or RecordingDispatcher()
        self._confidence_threshold = confidence_threshold

    @property
    def on_failure(self) -> "EscalationPolicy | None":
        """The failure-escalation policy this pipeline halts-and-escalates with,
        or ``None`` when a failure halts without dispatch."""
        return self._on_failure

    def add(
        self,
        agent: Agent,
        *,
        inputs: dict[str, From] | None = None,
        depends_on: list[str] | None = None,
        evidence: EvidencePolicy | None = None,
        when: "When | None" = None,
    ) -> "Pipeline":
        """Add an agent to the pipeline. Runs the registration-time static check
        on any declared bindings before accepting the step.

        ``evidence`` declares a per-step evidence-compression policy. It only
        affects a model-backed step (``model is not None``) with an enabled policy;
        deterministic and policy-less steps are unchanged.

        ``when`` declares an optional branch condition. The node is only run when
        the condition evaluates to true against the outputs produced so far. Static
        checks run at registration time.

        A ``JoinNode`` is also accepted here and routed to ``_add_join``; agents and
        joins share one node namespace.
        """
        from .graph import JoinNode

        if isinstance(agent, JoinNode):
            return self._add_join(agent)
        if agent.name in self._by_name:
            raise ValueError(
                f"pipeline '{self.name}' already has a node named '{agent.name}'; "
                f"agent and join names share one namespace"
            )
        inputs = inputs or {}
        if depends_on is None:
            depends_on = sorted({src.agent for src in inputs.values()})
        self._static_check(agent, inputs)
        self._static_check_when(agent, when)
        for tool_ref in agent.spec.tools:
            if tool_ref == EVIDENCE_TOOL_REF:
                # The evidence retrieval tool is auto-bound at run time when
                # evidence_store is provided to Pipeline.run; skip the
                # registration-time check so the caller's single evidence_store=
                # argument is the only wiring point needed.
                continue
            if not self._registry.has(tool_ref):
                raise ToolAccessError(
                    f"{agent.name}: declared tool '{tool_ref}' is not registered"
                )
        step = Step(agent=agent, inputs=inputs, depends_on=depends_on, evidence=evidence, when=when)
        self.steps.append(step)
        self._by_name[agent.name] = step
        return self

    def _add_join(self, join: "JoinNode") -> "Pipeline":
        if join.name in self._by_name:
            raise ValueError(
                f"pipeline '{self.name}' already has a node named '{join.name}'; "
                f"agent and join names share one namespace"
            )
        # static checks: sources exist; selecting policies need schema-compatible sources
        for src in join.sources:
            if src not in self._by_name:
                raise SchemaCompatibilityError(
                    f"join '{join.name}' source '{src}' is not in the pipeline"
                )
        if join.policy in ("exactly_one", "first_by_priority"):
            for src in join.sources:
                src_out = self._node_output_model(src)
                check_compatibility(src_out, join.output)
        elif join.policy == "all_present":
            # The join builds its `output` from `inputs` bindings at runtime via
            # `build_input`. Validate those bindings at REGISTRATION time the same
            # way `_static_check` validates agent `From` bindings, so a bad
            # source/field fails closed here (SchemaCompatibilityError) instead of
            # raising a raw KeyError/AttributeError from `build_input` at run time —
            # which would escape the scheduler's `except JoinError`.
            target_fields = join.output.model_fields
            for tfield, src in join.inputs.items():
                if tfield not in target_fields:
                    raise SchemaCompatibilityError(
                        f"join '{join.name}': bound field '{tfield}' is not in "
                        f"{join.output.__name__}"
                    )
                if src.agent not in self._by_name:
                    raise SchemaCompatibilityError(
                        f"join '{join.name}': binding source agent '{src.agent}' is "
                        f"not in the pipeline"
                    )
                src_model = self._node_output_model(src.agent)
                if src.field is None:
                    source_annotation = src_model
                else:
                    if src.field not in src_model.model_fields:
                        raise SchemaCompatibilityError(
                            f"join '{join.name}': source field "
                            f"'{src.agent}.{src.field}' does not exist on "
                            f"{src_model.__name__}"
                        )
                    source_annotation = src_model.model_fields[src.field].annotation
                target_annotation = target_fields[tfield].annotation
                check_compatibility(source_annotation, target_annotation)
        self.steps.append(join)
        self._by_name[join.name] = join
        return self

    def _node_output_model(self, name: str) -> type[BaseModel]:
        """Output model for a node (agent step or join) by name."""
        from .graph import JoinNode

        node = self._by_name[name]
        return node.output if isinstance(node, JoinNode) else node.agent.spec.output

    def _node_name(self, node) -> str:
        """The registered name of a step or join node."""
        from .graph import JoinNode

        return node.name if isinstance(node, JoinNode) else node.agent.name

    def _static_check(self, agent: Agent, inputs: dict[str, From]) -> None:
        target_fields = agent.spec.input.model_fields
        if not inputs:
            return  # first step / linear default validated at runtime against initial/predecessor
        for tfield, src in inputs.items():
            if tfield not in target_fields:
                raise SchemaCompatibilityError(
                    f"{agent.name}: bound field '{tfield}' is not in "
                    f"{agent.spec.input.__name__}"
                )
            if src.agent not in self._by_name:
                raise SchemaCompatibilityError(
                    f"{agent.name}: binding source agent '{src.agent}' is not in "
                    f"the pipeline"
                )
            src_model = self._node_output_model(src.agent)
            if src.field is None:
                source_annotation = src_model
            else:
                if src.field not in src_model.model_fields:
                    raise SchemaCompatibilityError(
                        f"{agent.name}: source field '{src.agent}.{src.field}' "
                        f"does not exist on {src_model.__name__}"
                    )
                source_annotation = src_model.model_fields[src.field].annotation
            target_annotation = target_fields[tfield].annotation
            check_compatibility(source_annotation, target_annotation)
        for fname, finfo in target_fields.items():
            if finfo.is_required() and fname not in inputs:
                raise SchemaCompatibilityError(
                    f"{agent.name}: required input field '{fname}' is not bound"
                )

    def _static_check_when(self, agent: Agent, when: "When | None") -> None:
        """Validate a branch condition at registration time.

        Raises ``SchemaCompatibilityError`` if:
        - the referenced agent is not yet in the pipeline;
        - that agent itself carries a ``when`` (conservative rule: gating on a
          conditionally-run node is the common authoring mistake — require an
          unconditional upstream instead);
        - the referenced field does not exist on the source agent's output model.
        """
        if when is None:
            return
        src_step = self._by_name.get(when.agent)
        if src_step is None:
            raise SchemaCompatibilityError(
                f"{agent.name}: When condition references agent '{when.agent}' which is not in the pipeline"
            )
        # Gating on a conditionally-run node is the common authoring mistake.
        if getattr(src_step, "when", None) is not None:
            raise SchemaCompatibilityError(
                f"{agent.name}: When condition references '{when.ref}' but node '{when.agent}' itself "
                f"carries a 'when' (conditionally run); gate on an unconditional node instead"
            )
        src_model = self._node_output_model(when.agent)
        if when.field not in src_model.model_fields:
            raise SchemaCompatibilityError(
                f"{agent.name}: When condition field '{when.ref}' does not exist on {src_model.__name__}"
            )

    def _topology_fingerprint(self) -> str:
        """Stable SHA-256 over the node list (names + binding refs + when refs).
        Resume is valid only against the same topology."""
        import hashlib, json
        from .graph import JoinNode
        shape = []
        for s in self.steps:
            if isinstance(s, JoinNode):
                shape.append({
                    "join": s.name,
                    "sources": list(s.sources),
                    "policy": s.policy,
                    "inputs": {f: src.ref for f, src in sorted(s.inputs.items())},
                })
            else:
                shape.append({
                    "agent": s.agent.name,
                    "inputs": {f: src.ref for f, src in sorted(s.inputs.items())},
                    "when": s.when.ref if s.when is not None else None,
                })
        blob = json.dumps(shape, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()

    def _halt(
        self,
        outputs: dict[str, BaseModel],
        steps_run: int,
        escalations: list[EscalationPackage],
        *,
        step: str | None,
        reason: str,
        received: Any,
        attempted_output: Any | None,
        agent_id: str | None = None,
        code: str,
        approval_request: "ApprovalRequest | None" = None,
    ) -> "RunResult":
        """Terminate the run. With no escalation policy this is a plain halt.
        With a policy, build + dispatch the escalation package and report status
        ``"escalated"``."""
        if self._on_failure is None:
            return RunResult(
                "halted", outputs, steps_run,
                halted_at=step, reason=reason, escalations=escalations,
                halt_code=code, approval_request=approval_request,
            )
        package = build_escalation(
            step=step or "(input)", reason=reason, received=received,
            attempted_output=attempted_output, trace=tuple(outputs.keys()),
            agent_id=agent_id,
        )
        self._dispatcher.dispatch(package, self._on_failure)
        return RunResult(
            "escalated", outputs, steps_run,
            halted_at=step, reason=reason, escalations=escalations + [package],
            halt_code=code, approval_request=approval_request,
        )

    def _dispatch_review(
        self,
        outputs: dict[str, BaseModel],
        policy: EscalationPolicy,
        *,
        step: str,
        reason: str,
        received: Any,
        attempted_output: Any | None,
    ) -> EscalationPackage:
        """Async review: build + dispatch a package and return it (the run continues)."""
        package = build_escalation(
            step=step, reason=reason, received=received,
            attempted_output=attempted_output, trace=tuple(outputs.keys()),
        )
        self._dispatcher.dispatch(package, policy)
        return package

    def _evaluate_join(self, node, present, outputs):
        from drawbore.schema import validate
        from .graph import JoinError
        if node.policy == "exactly_one":
            if len(present) != 1:
                raise JoinError(f"exactly_one over {node.sources}: {len(present)} ran")
            src = present[0]
            return src, validate(node.output, outputs[src].model_dump())
        if node.policy == "first_by_priority":
            for src in node.sources:
                if src in present:
                    return src, validate(node.output, outputs[src].model_dump())
            raise JoinError(f"first_by_priority over {node.sources}: none ran")
        # all_present
        missing = [s for s in node.sources if s not in present]
        if missing:
            raise JoinError(f"all_present over {node.sources}: missing {missing}")
        payload = build_input(node.inputs, outputs)
        return "all", validate(node.output, payload)

    async def run(
        self,
        initial: BaseModel,
        engine: OrchestratorEngine | None = None,
        *,
        run_id: str | None = None,
        checkpoints: CheckpointStore | None = None,
        identities: IdentityRegistry | None = None,
        tenant_id: str | None = None,
        audit: AuditSink | None = None,
        evidence_store: EvidenceStore | None = None,
        registry_override: Any | None = None,
        initial_trust: TrustLabel = TrustLabel.TRUSTED,
        approval: "ApprovalDecision | None" = None,
        effect_ledger: "EffectLedger | None" = None,
    ) -> RunResult:
        """Execute the pipeline and produce an audit record.

        The audit record is ALWAYS built and attached to ``RunResult.audit_trace``;
        when an ``audit`` sink is supplied it is also written there. See
        ``_run_inner`` for the execution semantics (halt-and-escalate, etc.).
        """
        if not isinstance(initial, BaseModel):
            raise TypeError(
                "Pipeline.run() requires a Pydantic BaseModel instance for "
                f"'initial', got {type(initial).__name__}"
            )
        if checkpoints is not None and run_id is None:
            raise ValueError(
                "Pipeline.run() with checkpoints= requires an explicit run_id: "
                "resume is keyed by the run identity, and a guessed default "
                "could silently resume a previous run's outputs. Pass run_id='...'."
            )
        if approval is not None and (checkpoints is None or run_id is None):
            raise ValueError(
                "Pipeline.run() with approval= requires checkpoints= and run_id=: "
                "a decision can only apply to a stored request from a resumable run."
            )
        resolved_run_id = run_id or f"{self.name}-run"
        recorder = AuditRecorder(
            run_id=resolved_run_id, pipeline=self.name,
            version=self.version, tenant_id=tenant_id,
        )
        ledger_builder = ResumeLedgerBuilder(run_id=resolved_run_id)
        result: RunResult | None = None
        try:
            result = await self._run_inner(
                initial, engine,
                run_id=resolved_run_id, checkpoints=checkpoints,
                identities=identities, tenant_id=tenant_id, recorder=recorder,
                ledger_builder=ledger_builder,
                evidence_store=evidence_store,
                registry_override=registry_override,
                initial_trust=initial_trust,
                approval=approval,
                effect_ledger=effect_ledger,
            )
            return result
        finally:
            # `finally` runs after the return value is evaluated but before the
            # frame returns to the caller; because RunResult is mutable, setting
            # result.audit_trace here IS visible to the caller. This is the
            # deliberate reason for the try/finally shape: every early-return halt
            # path inside _run_inner gets its audit record finalized in one place.
            if result is not None:
                record = recorder.build(
                    status=result.status, reason=result.reason,
                    halted_at=result.halted_at, escalations=len(result.escalations),
                )
                if audit is not None:
                    audit.write(record)
                result.audit_trace = record
                result.run_id = resolved_run_id
                result.resume_ledger = ledger_builder.build()
                result.metrics = recorder.build_metrics()

    async def _run_inner(
        self,
        initial: BaseModel,
        engine: OrchestratorEngine | None = None,
        *,
        run_id: str,
        checkpoints: CheckpointStore | None = None,
        identities: IdentityRegistry | None = None,
        tenant_id: str | None = None,
        recorder: AuditRecorder,
        ledger_builder: ResumeLedgerBuilder,
        evidence_store: EvidenceStore | None = None,
        registry_override: Any | None = None,
        initial_trust: TrustLabel = TrustLabel.TRUSTED,
        approval: "ApprovalDecision | None" = None,
        effect_ledger: "EffectLedger | None" = None,
    ) -> RunResult:
        """Execute the pipeline (halt-and-escalate default).

        Every failure halts; when an ``on_failure`` policy is configured the halt
        becomes a dispatched escalation (status ``"escalated"``). After a step
        succeeds, two triggers may fire: confidence below the declared threshold
        (honours the policy's sync/async mode) and an explicit
        ``requires_human_approval`` gate (always synchronous).

        Note: ``RunResult.escalations`` lists escalations raised during this
        invocation; on a resumed run an async escalation delivered in a prior
        attempt is not re-surfaced (its step is skipped as already completed).
        """
        engine = engine or LocalEngine()
        # A test-mode scoped registry overlay, when supplied, is the SINGLE registry
        # the whole run sees — it must feed both the ToolProxy and the ToolLoopBundle,
        # or mocked handlers/operations/loop tool schemas diverge.
        registry = registry_override if registry_override is not None else self._registry
        # Auto-bind the evidence retrieval handler when a run-time store is provided
        # and the tool is declared by at least one step but has no handler in the
        # run registry yet.  A per-run overlay is built to avoid mutating the shared
        # pipeline registry across runs.
        if evidence_store is not None and not registry.has(EVIDENCE_TOOL_REF):
            from .graph import JoinNode as _JN
            if any(
                EVIDENCE_TOOL_REF in s.agent.spec.tools
                for s in self.steps
                if not isinstance(s, _JN)
            ):
                _overlay = ToolRegistry()
                _overlay._tools.update(registry._tools)
                register_evidence_tool(_overlay, store=evidence_store)
                registry = _overlay
        # Join indices whose ledger `restored_join` entry was emitted during
        # pre-flight (in index order). The in-loop restore short-circuit must not
        # log them a second time.
        preflight_joins: set[int] = set()
        issuer = TokenIssuer()
        ledger = TaintLedger(initial_trust=initial_trust, managed=True)
        proxy = ToolProxy(registry, issuer, ledger=ledger, effect_ledger=effect_ledger)
        # Expose the proxy's per-call log (tool, operation, duration, disposition) on
        # RunResult.metrics via the recorder, which reads it at build time.
        recorder.bind_tool_log(proxy.log)
        state = RunState(run_id=run_id)
        if checkpoints is not None:
            from .graph import JoinNode as _JoinNode

            # Hand the store the live output model for each step so a durable
            # store can reconstruct typed outputs without reading a class path
            # off disk. The in-memory default ignores this (it keeps live
            # objects); a serialising store uses it for safe deserialisation.
            checkpoints.bind_models(run_id, {
                idx: (node_.output if isinstance(node_, _JoinNode)
                      else node_.agent.spec.output)
                for idx, node_ in enumerate(self.steps)
            })

            has_progress = any(
                checkpoints.is_completed(run_id, i) or checkpoints.is_skipped(run_id, i)
                for i in range(len(self.steps))
            )
            if has_progress:
                ledger_builder.set_resumed()
            fp = self._topology_fingerprint()
            if not checkpoints.fingerprint_matches(run_id, fp):
                if has_progress:
                    # Topology changed under a run with prior progress:
                    # index-keyed resume would mis-map and re-execution would
                    # double-fire completed steps. Refuse.
                    ledger_builder.set_topology("drifted")
                    return self._halt(
                        {}, 0, [],
                        step=self.name,
                        reason=(
                            "resume_drift: pipeline topology changed since "
                            f"run '{run_id}' was checkpointed; refusing to resume"
                        ),
                        received=None, attempted_output=None,
                        code="resume_drift",
                    )
                # A stale fingerprint with zero progress protects nothing:
                # treat as a fresh run and re-record.
                checkpoints.record_fingerprint(run_id, fp)
                ledger_builder.set_topology("recorded")
            else:
                # Skip the write when the fingerprint already matched and there
                # is prior progress: the stored value is identical, so writing
                # again is a no-op that durable stores still pay for per resume.
                # On fresh runs (not has_progress) the write records the initial
                # fingerprint, so it is always needed there.
                if not has_progress:
                    checkpoints.record_fingerprint(run_id, fp)
                ledger_builder.set_topology("verified" if has_progress else "recorded")

            if has_progress:
                # Pre-flight seal verification: every checkpoint-completed
                # agent step must match its stored seal BEFORE anything runs.
                refusals: list[tuple[int, str, tuple[str, ...]]] = []
                for i, node_ in enumerate(self.steps):
                    if isinstance(node_, _JoinNode):
                        # Joins are covered by the topology fingerprint and are
                        # never sealed, but a checkpoint-completed join still
                        # belongs in the ledger in index order — record it here
                        # so a refused resume lists it and a clean resume keeps
                        # entries ordered.
                        if checkpoints.is_completed(run_id, i):
                            ledger_builder.restored_join(i, node_.name)
                            preflight_joins.add(i)
                        continue
                    if not checkpoints.is_completed(run_id, i):
                        continue
                    current = seal_for(node_.agent.spec, node_.evidence, registry)
                    stored = checkpoints.seal_of(run_id, i)
                    if stored is None:
                        ledger_builder.refused(i, node_.agent.name, drifted_fields=())
                        refusals.append((i, node_.agent.name, ()))
                        continue
                    drifted = diff_seals(stored, current)
                    if drifted:
                        ledger_builder.refused(i, node_.agent.name, drifted_fields=drifted)
                        refusals.append((i, node_.agent.name, drifted))
                    else:
                        ledger_builder.restored(i, node_.agent.name, verified=True)
                if refusals:
                    idx0, name0, fields0 = refusals[0]
                    if fields0:
                        labels = ", ".join(field_label(f) for f in fields0)
                        detail = f"changed since checkpoint ({labels})"
                    else:
                        detail = (
                            "has a checkpointed output but no stored seal; "
                            "its semantics cannot be verified"
                        )
                    return self._halt(
                        {}, 0, [],
                        step=name0,
                        reason=f"resume_drift: step '{name0}' {detail}; refusing to resume",
                        received=None, attempted_output=None,
                        code="resume_drift",
                    )
        outputs: dict[str, BaseModel] = {}
        # Two taint surfaces: `ledger` holds the WITHIN-step mutable scope the
        # proxy gates/bumps during execution; `output_trust` is the CROSS-STEP
        # propagation map — each completed step's final label, used to seed
        # downstream scopes, gate When predicates, and persist via
        # checkpoints.record_trust.
        output_trust: dict[str, TrustLabel] = {}
        skipped: set[str] = set()
        escalations: list[EscalationPackage] = []
        steps_run = 0
        approval_consumed = False

        # The run's initial input is external — sanitize it before anything runs.
        try:
            sanitize(initial.model_dump())
        except SanitizationError as exc:
            return self._halt(
                outputs, 0, escalations,
                step=None, reason=f"sanitization: {exc}", received=None, attempted_output=None,
                code="sanitization",
            )

        executor = StepExecutor(
            proxy=proxy, issuer=issuer, registry=registry,
            engine=engine, evidence_store=evidence_store,
        )
        from .graph import JoinNode, JoinError
        for idx, node in enumerate(self.steps):
            name = node.name if isinstance(node, JoinNode) else node.agent.name
            state.step = idx

            if isinstance(node, JoinNode):
                # The checkpoint-resume short-circuit applies to a completed join
                # too: it restores its stored output (joins call step_succeeded on
                # success, so output_of returns the join value). Joins have no
                # identity gate, so this short-circuit is safe at the branch top.
                if checkpoints is not None and checkpoints.is_completed(run_id, idx):
                    outputs[name] = checkpoints.output_of(run_id, idx)
                    output_trust[name] = checkpoints.trust_of(run_id, idx)
                    if idx not in preflight_joins:
                        # Defensive fallback: pre-flight logs every completed join,
                        # so this branch cannot fire under the current pre-flight
                        # contract; it guards against future callers that skip it.
                        ledger_builder.restored_join(idx, name)
                    steps_run += 1
                    continue
                # joins are exempt from skip-propagation: always dispatch.
                present = [s for s in node.sources if s not in skipped and s in outputs]
                try:
                    selected, value = self._evaluate_join(node, present, outputs)
                except JoinError as exc:
                    return self._halt(
                        outputs, steps_run, escalations,
                        step=name, reason=f"join_policy_violation: {exc}",
                        received=None, attempted_output=None,
                        code="join_policy_violation",
                    )
                outputs[name] = value
                if selected == "all":
                    output_trust[name] = join(*(output_trust.get(s, TrustLabel.UNTRUSTED) for s in node.sources))
                else:
                    output_trust[name] = output_trust.get(selected, TrustLabel.UNTRUSTED)
                recorder.record_step(
                    index=idx, agent=name, version="", agent_id=None,
                    input_hash=None, output_hash=payload_hash(value.model_dump()),
                    tool_calls=(), node_kind="join",
                    join=f"{node.policy} over {node.sources} -> {selected}",
                )
                steps_run += 1
                ledger_builder.executed_join(idx, name)
                if checkpoints is not None:
                    checkpoints.step_succeeded(run_id, idx, value)
                    checkpoints.record_trust(run_id, idx, output_trust[name])
                continue

            step = node
            agent_id = identities.agent_id_of(name) if identities is not None else None

            # --- reachability: decide whether this node runs ---
            predecessor = self._node_name(self.steps[idx - 1]) if idx > 0 else None
            reach_sources = (
                set(src.agent for src in step.inputs.values()) if step.inputs
                else ({predecessor} if predecessor is not None else set())
            )
            # Replay a skip recorded in a prior attempt of this run:
            # do not silently re-derive it — the recorder is fresh on a resumed run,
            # so re-record the skip line for trace completeness.
            if checkpoints is not None and checkpoints.is_skipped(run_id, idx):
                skipped.add(name)
                recorder.record_skipped_step(
                    index=idx, agent=name, version=step.agent.spec.version,
                    agent_id=agent_id, condition="(resumed: skipped)",
                )
                ledger_builder.skipped(idx, name)
                continue

            cascade = reach_sources & skipped
            if cascade:
                skipped.add(name)
                if checkpoints is not None:
                    checkpoints.step_skipped(run_id, idx)
                recorder.record_skipped_step(
                    index=idx, agent=name, version=step.agent.spec.version,
                    agent_id=agent_id,
                    # Intentionally names the lexicographically-first of possibly
                    # several skipped sources — deterministic and legible for auditors.
                    condition=f"cascade: {sorted(cascade)[0]} skipped",
                )
                ledger_builder.skipped(idx, name)
                continue
            if step.when is not None:
                if step.when.agent in skipped:
                    # Reachable when a `when` gates (without binding) on a node that
                    # was cascade-skipped; fails closed, never silently skips.
                    return self._halt(
                        outputs, steps_run, escalations,
                        step=name, reason=f"condition_unevaluable: {step.when.agent} skipped",
                        received=None, attempted_output=None,
                        code="condition_unevaluable",
                    )
                # Branch-gate: a When condition that reads an UNTRUSTED output must
                # halt — the branch decision itself is tainted data. Checked after
                # the unevaluable guard so a missing-upstream still halts
                # condition_unevaluable (not condition_tainted).
                if output_trust.get(step.when.agent, TrustLabel.UNTRUSTED) is TrustLabel.UNTRUSTED:
                    return self._halt(
                        outputs, steps_run, escalations,
                        step=name,
                        reason=(
                            f"condition_tainted: {step.when.legible()} reads untrusted "
                            f"output from {step.when.agent}"
                        ),
                        received=None, attempted_output=None,
                        code="condition_tainted",
                    )
                taken = step.when.evaluate(outputs)
                if not taken:
                    skipped.add(name)
                    # Do NOT persist condition-false skips. On resume the
                    # checkpointed upstream output is restored before this node is
                    # reached, so the condition re-derives deterministically here
                    # AND the resumed trace preserves the ORIGINAL legible condition
                    # string (legibility-first). Only cascade skips are persisted.
                    recorder.record_skipped_step(
                        index=idx, agent=name, version=step.agent.spec.version,
                        agent_id=agent_id,
                        condition=f"{step.when.legible()} (false)",
                    )
                    ledger_builder.skipped(idx, name)
                    continue

            # Identity gate: a registered agent that is suspended, decommissioned,
            # or pending re-attestation is blocked from running and halts-and-
            # escalates. An unregistered (draft) agent is not gated here.
            if identities is not None:
                block = identities.run_block_reason(name)
                if block is not None:
                    return self._halt(
                        outputs, steps_run, escalations,
                        step=name, reason=f"identity_{block}",
                        received=None, attempted_output=None, agent_id=agent_id,
                        code="identity_blocked",
                    )

            if checkpoints is not None and checkpoints.is_completed(run_id, idx):
                outputs[name] = checkpoints.output_of(run_id, idx)
                output_trust[name] = checkpoints.trust_of(run_id, idx)
                steps_run += 1
                continue
            if checkpoints is not None:
                checkpoints.step_started(run_id, idx)

            # Seed the step's taint scope: join the trust labels of all input
            # sources. An initial step (no sources) inherits the run's initial_trust.
            if reach_sources:
                seed = join(*(output_trust.get(s, TrustLabel.UNTRUSTED) for s in reach_sources))
            else:
                seed = initial_trust
            ledger.seed(run_id, idx, seed)

            # Pre-execution handler: if this step has a pending approval request,
            # handle it before executing the agent (either re-surface the request
            # on a decision-less poll, or apply the decision on a decided resume).
            # This prevents a gated side-effecting agent from re-executing on a
            # decision-less poll (double-fire).
            if checkpoints is not None:
                pending = checkpoints.approval_request_of(run_id)
                if pending is not None and pending.step == name:
                    if approval is None:
                        # Decision-less resume: re-surface the pending request
                        # WITHOUT re-executing the gated agent (no double-fire).
                        return self._halt(
                            outputs, steps_run, escalations, step=name,
                            reason="requires_human_approval",
                            received=None, attempted_output=pending.proposed_output,
                            code="requires_human_approval", approval_request=pending,
                        )
                    # Decision supplied — apply it (existing id-check / reject /
                    # amend-or-approve / schema-gate / record / full-success-block).
                    if approval.request_id != pending.request_id:
                        return self._halt(
                            outputs, steps_run, escalations, step=name,
                            reason=(
                                "approval_error: decision is bound to request "
                                f"'{approval.request_id}' but the pending request "
                                f"is '{pending.request_id}'"
                            ),
                            received=None, attempted_output=None,
                            code="approval_error",
                        )
                    if approval.verdict == "rejected":
                        checkpoints.clear_approval_request(run_id)
                        approval_consumed = True
                        return self._halt(
                            outputs, steps_run, escalations, step=name,
                            reason=(
                                "approval_rejected: reviewer "
                                f"'{approval.reviewer_id}' rejected the proposed output"
                                + (f" — {approval.rationale}" if approval.rationale else "")
                            ),
                            received=None, attempted_output=pending.proposed_output,
                            code="approval_rejected",
                        )
                    chosen = (
                        approval.amended_output if approval.verdict == "amended"
                        else pending.proposed_output
                    )
                    try:
                        validated_out = validate(step.agent.spec.output, chosen)
                    except SchemaValidationError as exc:
                        checkpoints.clear_approval_request(run_id)
                        approval_consumed = True
                        return self._halt(
                            outputs, steps_run, escalations, step=name,
                            reason=f"schema_violation: {exc}",
                            received=None, attempted_output=chosen,
                            code="schema_violation",
                        )
                    checkpoints.clear_approval_request(run_id)
                    approval_consumed = True
                    original_hash = payload_hash(pending.proposed_output)
                    applied_hash = payload_hash(validated_out.model_dump())
                    recorder.record_step(
                        index=idx, agent=name, version=step.agent.spec.version,
                        agent_id=agent_id, input_hash=None, output_hash=applied_hash,
                        tool_calls=(),
                        human_decision=approval.verdict, reviewer_id=approval.reviewer_id,
                        amendment_original_hash=(
                            original_hash if approval.verdict == "amended" else None
                        ),
                        amendment_applied_hash=(
                            applied_hash if approval.verdict == "amended" else None
                        ),
                    )
                    outputs[name] = validated_out
                    output_trust[name] = join(TrustLabel(pending.proposed_output_trust), ledger.scope(run_id, idx))
                    checkpoints.step_succeeded(run_id, idx, validated_out)
                    checkpoints.record_trust(run_id, idx, output_trust[name])
                    checkpoints.record_seal(run_id, idx, seal_for(step.agent.spec, step.evidence, registry))
                    ledger_builder.executed(idx, name, sealed=True)
                    steps_run += 1
                    continue

            payload = build_input(
                step.inputs, outputs, initial=initial,
                predecessor=self._node_name(self.steps[idx - 1]) if idx > 0 else None,
            )
            step_start = time.monotonic()
            outcome = await executor.execute(
                agent=step.agent, idx=idx, run_id=run_id, payload=payload,
                evidence_policy=step.evidence, tenant_id=tenant_id, agent_id=agent_id,
            )
            step_duration = time.monotonic() - step_start
            if isinstance(outcome, Halt):
                state.error_count += 1
                # If the failing step ran tools (e.g. a denied in-loop call),
                # record a FAILED step so the audit trail shows what it did — the
                # SAME rendering the success path uses, so a denied call reads
                # identically. Pure agent/schema errors (no tool calls) are
                # unaffected: tool_calls is empty and no record is added.
                if outcome.audit.tool_calls:
                    recorder.record_failed_step(
                        index=idx, agent=name, version=step.agent.spec.version,
                        agent_id=agent_id, input_hash=outcome.audit.input_hash,
                        tool_calls=outcome.audit.tool_calls, reason=outcome.reason,
                        model_turns=outcome.audit.model_turns,
                        reprompts=outcome.audit.reprompts, model=None,
                        duration_seconds=step_duration,
                        tokens=outcome.audit.tokens, cost=outcome.audit.cost,
                    )

                return self._halt(
                    outputs, steps_run, escalations,
                    step=name, reason=outcome.reason,
                    received=outcome.received, attempted_output=outcome.attempted,
                    code=outcome.code,
                )

            # Step-end orphan check: a clean step that consumed FEWER effectful
            # tool calls than were recorded for it (the resumed run skipped an
            # already-fired effect) cannot be proven exactly-once — halt fail-
            # closed. Only runs on a clean (non-halted) step with a ledger wired;
            # a halted step legitimately leaves recorded entries to re-replay on
            # the next resume, so it is excluded by the early return above.
            if effect_ledger is not None:
                consumed = proxy.effects_consumed(run_id, idx)
                recorded = effect_ledger.recorded_count(run_id, idx)
                if recorded > consumed:
                    orphans = effect_ledger.entries_from(run_id, idx, consumed)
                    detail = ", ".join(
                        f"position {e.position} ({e.tool_ref})" for e in orphans
                    )
                    return self._halt(
                        outputs, steps_run, escalations,
                        step=name,
                        reason=(
                            f"effect_divergence: step '{name}' recorded {recorded} "
                            f"effect(s) but the resumed run consumed {consumed}; "
                            f"orphaned recorded effect(s) not re-fired: {detail}"
                        ),
                        received=None, attempted_output=None,
                        code="effect_divergence",
                    )

            validated_out = outcome.output
            audit = outcome.audit

            # --- post-success escalation triggers ---
            # Confidence below the declared threshold. Only models that explicitly
            # inherit HasConfidence are checked.
            if self._confidence_threshold is not None and isinstance(validated_out, HasConfidence):
                confidence = getattr(validated_out, "confidence", None)
                if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                    # A HasConfidence marker with no usable numeric `confidence`
                    # field is a misconfiguration — fail closed (halt/escalate),
                    # never let the AttributeError propagate out of run().
                    return self._halt(
                        {**outputs, name: validated_out}, steps_run + 1, escalations,
                        step=name,
                        reason=(
                            "confidence_marker_without_value: output is marked "
                            "HasConfidence but has no numeric 'confidence' field"
                        ),
                        received=outcome.validated_input, attempted_output=validated_out,
                        code="confidence_marker_without_value",
                    )
                if confidence < self._confidence_threshold:
                    reason = (
                        f"confidence_below_threshold: "
                        f"{confidence} < {self._confidence_threshold}"
                    )
                    if self._on_failure is not None and self._on_failure.mode == "async":
                        escalations.append(self._dispatch_review(
                            {**outputs, name: validated_out},
                            self._on_failure,
                            step=name, reason=reason,
                            received=outcome.validated_input, attempted_output=validated_out,
                        ))
                        # async review: fall through and continue the run.
                    else:
                        return self._halt(
                            {**outputs, name: validated_out}, steps_run + 1, escalations,
                            step=name, reason=reason,
                            received=outcome.validated_input, attempted_output=validated_out,
                            code="confidence_below_threshold",
                        )

            # Explicit human-approval gate — always a synchronous gate.
            # The pre-execution handler above owns all reuse logic; this block
            # always mints a fresh request (a second mint on a store-less run
            # is info-only — no decision can arrive without a store).
            if step.agent.spec.requires_human_approval:
                request = ApprovalRequest(
                    request_id=uuid.uuid4().hex, run_id=run_id, step=name,
                    question=f"Approve the proposed output of step '{name}'?",
                    reason="requires_human_approval",
                    package_legible=build_escalation(
                        step=name, reason="requires_human_approval",
                        received=outcome.validated_input,
                        attempted_output=validated_out,
                        trace=tuple(outputs.keys()), agent_id=agent_id,
                    ).legible(),
                    proposed_output=validated_out.model_dump(mode="json"),
                    proposed_output_trust=ledger.scope(run_id, idx).value,
                )
                if checkpoints is not None:
                    checkpoints.record_approval_request(run_id, request)
                return self._halt(
                    {**outputs, name: validated_out}, steps_run + 1, escalations,
                    step=name, reason="requires_human_approval",
                    received=outcome.validated_input, attempted_output=validated_out,
                    code="requires_human_approval",
                    approval_request=request,
                )

            recorder.record_step(
                index=idx, agent=name, version=step.agent.spec.version, agent_id=agent_id,
                input_hash=audit.input_hash,
                output_hash=payload_hash(validated_out.model_dump()),
                tool_calls=audit.tool_calls, evidence=audit.evidence_summary,
                model_turns=audit.model_turns, reprompts=audit.reprompts,
                model=audit.model_audit,
                condition=(f"{step.when.legible()} (true)" if step.when is not None else None),
                duration_seconds=step_duration,
                tokens=audit.tokens, cost=audit.cost,
            )
            outputs[name] = validated_out
            output_trust[name] = ledger.scope(run_id, idx)
            if checkpoints is not None:
                checkpoints.step_succeeded(run_id, idx, validated_out)
                checkpoints.record_trust(run_id, idx, output_trust[name])
                checkpoints.record_seal(run_id, idx, seal_for(step.agent.spec, step.evidence, registry))
                ledger_builder.executed(idx, name, sealed=True)
            else:
                ledger_builder.executed(idx, name, sealed=False)
            steps_run += 1

        if approval is not None and not approval_consumed:
            return self._halt(
                outputs, steps_run, escalations, step="",
                reason=(
                    "approval_error: an approval decision was supplied but no "
                    "pending request consumed it"
                ),
                received=None, attempted_output=None, code="approval_error",
            )
        return RunResult("completed", outputs, steps_run, escalations=escalations)

    def test_mode(
        self,
        *,
        mock_tools: "Mapping[str, Any] | None" = None,
        mock_model_responses: "Mapping[str, Any] | None" = None,
        mock_loop_scripts: "Mapping[str, Any] | None" = None,
        mock_model_usage: "Mapping[str, Any] | None" = None,
        allow_real_tools: "Collection[str]" = (),
        evidence_store: "Any | None" = None,
        audit_sink: "Any | None" = None,
        run_id_prefix: str | None = None,
        max_llm_calls: int = 8,
        llm_config: "Any | None" = None,
        credential_checker: "Any | None" = None,
    ):
        """Open a local test harness around this pipeline.

        Returns an async context manager: ``async with pipeline.test_mode(...) as
        test: result = await test.run(initial)``. Mocks fake EXTERNAL effects only —
        every Drawbore safety control runs for real through ``Pipeline.run``. See
        ``drawbore.testing`` for the mock argument shapes.
        """
        from drawbore.testing import TestMode

        return TestMode(
            self,
            mock_tools=mock_tools,
            mock_model_responses=mock_model_responses,
            mock_loop_scripts=mock_loop_scripts,
            mock_model_usage=mock_model_usage,
            allow_real_tools=allow_real_tools,
            evidence_store=evidence_store,
            audit_sink=audit_sink,
            run_id_prefix=run_id_prefix,
            max_llm_calls=max_llm_calls,
            llm_config=llm_config,
            credential_checker=credential_checker,
        )
