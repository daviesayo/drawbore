"""Builds an AuditRecord during a run.

The pipeline records each successful step's detail as it goes, then calls
``build`` once at the end with the run's final status/reason/escalation count. The
recorder is the only place that knows how the summary counts are derived: ``steps``
= successful steps recorded; ``schema_violations`` = 1 when the run halted on a
schema-validation reason (a run halts at the first violation, so at most one per
run); ``escalations`` = the count the pipeline passes in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .metrics import RunMetrics, StepMetric, ToolCallMetric
from .records import AuditRecord, StepAuditRecord

if TYPE_CHECKING:
    from drawbore.llm import ModelAudit, TokenUsage


class AuditRecorder:
    def __init__(self, *, run_id: str, pipeline: str, version: str, tenant_id: str | None) -> None:
        self._run_id = run_id
        self._pipeline = pipeline
        self._version = version
        self._tenant_id = tenant_id
        self._steps: list[StepAuditRecord] = []
        # Parallel quantitative view (duration/tokens/cost per executed step) plus a
        # bound reference to the tool-proxy log (read at build time). Kept separate
        # from the legible audit record so the immutable trace is unaffected.
        self._step_metrics: list[StepMetric] = []
        self._tool_log: list[dict] | None = None

    def bind_tool_log(self, log: list[dict]) -> None:
        """Bind the live tool-proxy log (a list mutated in place during the run).
        ``build_metrics`` reads it at the end, so binding the reference once is
        enough."""
        self._tool_log = log

    def record_step(
        self,
        *,
        index: int,
        agent: str,
        version: str,
        agent_id: str | None,
        input_hash: str | None,
        output_hash: str | None,
        tool_calls: tuple[str, ...],
        evidence: str | None = None,
        model_turns: int = 0,
        reprompts: int = 0,
        model: "ModelAudit | None" = None,
        condition: str | None = None,
        node_kind: str = "agent",
        join: str | None = None,
        duration_seconds: float | None = None,
        tokens: "TokenUsage | None" = None,
        cost: float | None = None,
        human_decision: str | None = None,
        reviewer_id: str | None = None,
        amendment_original_hash: str | None = None,
        amendment_applied_hash: str | None = None,
    ) -> None:
        """Record one successfully-completed step."""
        self._steps.append(
            StepAuditRecord(
                index=index, agent=agent, version=version, agent_id=agent_id,
                status="ok", input_hash=input_hash, output_hash=output_hash,
                tool_calls=tuple(tool_calls), reason=None, evidence=evidence,
                model_turns=model_turns, reprompts=reprompts, model=model,
                condition=condition, node_kind=node_kind, join=join,
                human_decision=human_decision, reviewer_id=reviewer_id,
                amendment_original_hash=amendment_original_hash,
                amendment_applied_hash=amendment_applied_hash,
            )
        )
        self._step_metrics.append(
            StepMetric(
                index=index, agent=agent, duration_seconds=duration_seconds,
                tokens=tokens, cost=cost,
            )
        )

    def record_failed_step(
        self,
        *,
        index: int,
        agent: str,
        version: str,
        agent_id: str | None,
        input_hash: str | None,
        tool_calls: tuple[str, ...],
        reason: str,
        model_turns: int = 0,
        reprompts: int = 0,
        model: "ModelAudit | None" = None,
        duration_seconds: float | None = None,
        tokens: "TokenUsage | None" = None,
        cost: float | None = None,
    ) -> None:
        """Record a step that halted AFTER doing work (e.g. a denied in-loop tool
        call). Status ``"failed"``; NOT counted in the run's ``steps`` (success
        only) but present in ``step_records`` so the trail shows what it did."""
        self._steps.append(
            StepAuditRecord(
                index=index, agent=agent, version=version, agent_id=agent_id,
                status="failed", input_hash=input_hash, output_hash=None,
                tool_calls=tuple(tool_calls), reason=reason, model_turns=model_turns,
                reprompts=reprompts, model=model,
            )
        )
        self._step_metrics.append(
            StepMetric(
                index=index, agent=agent, duration_seconds=duration_seconds,
                tokens=tokens, cost=cost,
            )
        )

    def record_skipped_step(
        self,
        *,
        index: int,
        agent: str,
        version: str,
        agent_id: str | None,
        condition: str | None,
    ) -> None:
        """Record a node that did not run (branch not taken / cascade). Status
        "skipped"; NOT counted in `steps`; present in `step_records` so absence is
        explained (legibility-first)."""
        self._steps.append(
            StepAuditRecord(
                index=index, agent=agent, version=version, agent_id=agent_id,
                status="skipped", input_hash=None, output_hash=None,
                tool_calls=(), reason=None, condition=condition,
            )
        )

    def build(self, *, status: str, reason: str | None, halted_at: str | None, escalations: int) -> AuditRecord:
        """Finalize the record for a finished run."""
        # The executor emits the stable token "schema_violation: ..."; match
        # the token, never prose. A run halts at the first violation, so the count
        # is at most 1 per run.
        schema_violations = 1 if (reason is not None and reason.startswith("schema_violation:")) else 0
        return AuditRecord(
            run_id=self._run_id,
            pipeline=self._pipeline,
            version=self._version,
            tenant_id=self._tenant_id,
            status=status,
            steps=sum(1 for s in self._steps if s.status == "ok"),
            escalations=escalations,
            schema_violations=schema_violations,
            reason=reason,
            step_records=tuple(self._steps),
            halted_at=halted_at,
        )

    def build_metrics(self) -> RunMetrics:
        """Finalize the quantitative metrics for a finished run: the per-step
        durations/tokens/cost accumulated during the run plus the tool-proxy call log
        (empty when no tools ran or the log was never bound)."""
        tool_calls = (
            tuple(ToolCallMetric.from_log_entry(e) for e in self._tool_log)
            if self._tool_log is not None
            else ()
        )
        return RunMetrics(
            run_id=self._run_id,
            steps=tuple(self._step_metrics),
            tool_calls=tool_calls,
        )
