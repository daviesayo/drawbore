"""StepExecutor — run ONE agent step and return a typed Outcome.

Lifted out of Pipeline._run_inner so the scheduler stays thin and every workflow
construct (branch/skip/join) composes against one execution unit. Owns no
cross-step state: it is handed the shared proxy/issuer/registry and returns the
audit payload the scheduler records.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from opentelemetry.trace import Status, StatusCode

from drawbore.agent import Agent
from drawbore.errors import halt_reason_for
from drawbore.evidence import (
    EvidencePolicy,
    EvidenceStore,
    InMemoryEvidenceStore,
    compress_for_model,
)
from drawbore.observability import genai_span, payload_hash, semconv
from drawbore.orchestration import StepExecution, ToolLoopBundle
from drawbore.schema import validate
from drawbore.schema.errors import SchemaValidationError
from drawbore.tools import (
    RunContext,
    build_tool_context,
    reset_run_context,
    set_run_context,
)

from .outcome import Halt, Ok, Outcome, StepAudit


def _render_schema_errors(errors: list[dict]) -> str:
    """Render Pydantic error dicts as a regulator-readable summary
    ("field: message; ..."), never a raw dict dump with error URLs."""
    parts = []
    for e in errors:
        loc = ".".join(str(p) for p in e.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {e.get('msg', 'invalid')}")
    return "; ".join(parts) if parts else "validation failed"


def _passthrough_after_invalid(decision):
    """Demote a compressed decision to passthrough when its view failed
    re-validation and the policy is non-strict — the model gets the original."""
    return replace(
        decision, decision="passthrough",
        reason="compressed view failed re-validation; passthrough", handle_id=None,
    )


class StepExecutor:
    """Runs one agent step against the shared proxy/issuer/registry/engine."""

    def __init__(self, *, proxy, issuer, registry, engine, evidence_store: EvidenceStore | None):
        self._proxy = proxy
        self._issuer = issuer
        self._registry = registry
        self._engine = engine
        self._evidence_store = evidence_store

    def _tool_calls_since(self, start: int) -> tuple[str, ...]:
        return tuple(
            f"{e['tool']} ({e['operation']}) -> {e['result']}"
            for e in self._proxy.log[start:]
        )

    async def execute(
        self,
        *,
        agent: Agent,
        idx: int,
        run_id: str,
        payload: Any,
        evidence_policy: EvidencePolicy | None,
        tenant_id: str | None,
        agent_id: str | None,
    ) -> Outcome:
        spec = agent.spec
        name = spec.name

        try:
            validated_in = validate(spec.input, payload)
        except SchemaValidationError as exc:
            return Halt(
                reason=f"schema_violation: input: {_render_schema_errors(exc.errors)}",
                received=payload, attempted=None, audit=StepAudit(input_hash=None),
                code="schema_violation",
            )

        # --- pre-model evidence compression ---
        model_in = validated_in
        evidence_summary: str | None = None
        evidence_attrs: dict[str, Any] = {}
        policy = evidence_policy
        if policy is not None and policy.enabled and spec.model is not None:
            store = self._evidence_store if self._evidence_store is not None else InMemoryEvidenceStore()
            try:
                view, decision, handle = compress_for_model(
                    validated_in.model_dump(), policy, store=store,
                    run_id=run_id, step=idx, source_agent=name,
                )
            except Exception as exc:
                code = halt_reason_for(exc)
                return Halt(
                    reason=f"{code}: {exc}",
                    received=validated_in, attempted=None,
                    audit=StepAudit(input_hash=payload_hash(validated_in.model_dump())),
                    code=code,
                )
            if decision.decision == "compressed" and handle is not None:
                store.set_policy(
                    handle.handle_id,
                    allow_full=policy.allow_full_retrieval,
                    allow_search=policy.allow_search_retrieval,
                )
                try:
                    model_in = validate(spec.input, view)
                except SchemaValidationError:
                    if policy.strict:
                        return Halt(
                            reason="evidence_error: compressed view failed re-validation",
                            received=validated_in, attempted=None,
                            audit=StepAudit(input_hash=payload_hash(validated_in.model_dump())),
                            code="evidence_error",
                        )
                    model_in = validated_in
                    store.delete(handle.handle_id)
                    decision = _passthrough_after_invalid(decision)
            evidence_summary = decision.legible()
            evidence_attrs = {
                semconv.DRAWBORE_EVIDENCE_DECISION: decision.decision,
                semconv.DRAWBORE_EVIDENCE_TRANSFORM: decision.transform,
                semconv.DRAWBORE_EVIDENCE_HANDLE_ID: decision.handle_id,
                semconv.DRAWBORE_EVIDENCE_ORIGINAL_TOKENS: decision.original_tokens,
                semconv.DRAWBORE_EVIDENCE_COMPRESSED_TOKENS: decision.compressed_tokens,
                semconv.DRAWBORE_EVIDENCE_POLICY: decision.policy,
            }

        tools = (
            build_tool_context(spec.tools, self._proxy, self._issuer)
            if spec.tools else None
        )
        input_hash = payload_hash(model_in.model_dump())
        agent_attrs = {
            semconv.GEN_AI_AGENT_NAME: name,
            semconv.GEN_AI_AGENT_VERSION: spec.version,
            semconv.DRAWBORE_RISK_TIER: spec.risk_tier,
            semconv.DRAWBORE_RUN_ID: run_id,
            semconv.DRAWBORE_STEP: idx,
            semconv.DRAWBORE_TENANT_ID: tenant_id,
            semconv.DRAWBORE_INPUT_HASH: input_hash,
            semconv.GEN_AI_AGENT_ID: agent_id,
            **evidence_attrs,
        }
        tool_loop = None
        if spec.model is not None and spec.tools:
            tool_loop = ToolLoopBundle(
                proxy=self._proxy, issuer=self._issuer, registry=self._registry,
                declared=spec.tools, run_ctx=RunContext(run_id=run_id, step=idx),
            )
        ctx_token = set_run_context(RunContext(run_id=run_id, step=idx))
        tool_log_start = len(self._proxy.log)
        try:
            with genai_span(semconv.OP_INVOKE_AGENT, name, agent_attrs) as span:
                raw = await self._engine.run_step(spec, model_in, tools=tools, tool_loop=tool_loop)
                span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            code = halt_reason_for(exc)
            tool_calls = self._tool_calls_since(tool_log_start)
            return Halt(
                reason=f"{code}: {exc}",
                received=validated_in, attempted=None,
                audit=StepAudit(
                    input_hash=input_hash, tool_calls=tool_calls,
                    model_turns=(len(tool_loop.turns) if tool_loop is not None else 0),
                ),
                code=code,
            )
        finally:
            reset_run_context(ctx_token)

        execution = raw if isinstance(raw, StepExecution) else StepExecution(output=raw)
        try:
            validated_out = validate(spec.output, execution.output)
        except SchemaValidationError as exc:
            tool_calls = self._tool_calls_since(tool_log_start)
            return Halt(
                reason=f"schema_violation: output: {_render_schema_errors(exc.errors)}",
                received=validated_in, attempted=execution.output,
                audit=StepAudit(input_hash=input_hash, tool_calls=tool_calls,
                                model_turns=execution.model_turns),
                code="schema_violation",
            )

        tool_calls = self._tool_calls_since(tool_log_start)
        return Ok(
            output=validated_out,
            validated_input=validated_in,
            audit=StepAudit(
                input_hash=input_hash, tool_calls=tool_calls,
                evidence_summary=evidence_summary,
                model_audit=execution.model_audit, model_turns=execution.model_turns,
            ),
        )
