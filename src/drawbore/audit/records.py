"""Audit records — the regulator-legible trace of a run.

A separate concern from operational observability (OTel spans): an immutable,
append-only, human-readable record. The MIT core ships the basic, readable,
exportable log; tamper-evidence, cryptographic signing, compliance-format export,
and SIEM integration are managed-service/enterprise and are deliberately NOT here.
Legibility-first: ``legible()`` must read without an engineer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drawbore.llm import ModelAudit


@dataclass(frozen=True)
class StepAuditRecord:
    """What one step did: which agent (and version/id) ran, the outcome, the
    input/output payload hashes (comparable to the step's spans), and the
    tool calls it made (each rendered ``"<tool> (<operation>) -> <result>"``)."""

    index: int
    agent: str
    version: str
    agent_id: str | None
    status: str
    input_hash: str | None
    output_hash: str | None
    tool_calls: tuple[str, ...]
    reason: str | None = None
    evidence: str | None = None
    model_turns: int = 0
    reprompts: int = 0
    model: "ModelAudit | None" = None
    condition: str | None = None
    node_kind: str = "agent"
    join: str | None = None


@dataclass(frozen=True)
class AuditRecord:
    """The append-only record of a single run. ``steps`` is the number of steps
    that completed successfully; ``escalations`` and ``schema_violations`` are
    summary counts; ``step_records`` is the per-step detail. The record's
    ``steps``, ``escalations``, and ``schema_violations`` fields are the counts
    a compliance reviewer checks first."""

    run_id: str
    pipeline: str
    version: str
    tenant_id: str | None
    status: str
    steps: int
    escalations: int
    schema_violations: int
    reason: str | None
    step_records: tuple[StepAuditRecord, ...]
    halted_at: str | None = None

    def legible(self) -> str:
        """Render the run as a trace a compliance officer can read end-to-end."""
        header = f"Run '{self.run_id}' of pipeline '{self.pipeline}' v{self.version}"
        if self.tenant_id is not None:
            header += f" (tenant {self.tenant_id})"
        lines = [
            f"{header}: {self.status}.",
            f"Steps completed: {self.steps}.",
            f"Escalations: {self.escalations}.",
            f"Schema violations: {self.schema_violations}.",
        ]
        for s in self.step_records:
            if s.node_kind == "join":
                # A join has no agent version — rendering `v-`/`v` would be a
                # misleading version token, so omit it (legibility-first).
                who = f"step {s.index} '{s.agent}' (join)"
            else:
                who = f"step {s.index} '{s.agent}' v{s.version}"
            if s.agent_id is not None:
                who += f" (agent id {s.agent_id})"
            line = f"- {who}: {s.status}"
            if s.condition is not None:
                line += f"; condition: {s.condition}"
            if s.tool_calls:
                line += f"; tools: {', '.join(s.tool_calls)}"
            if s.evidence is not None:
                line += f"; {s.evidence}"
            if s.model is not None:
                line += f"; {s.model.legible()}"
            if s.model_turns:
                line += f"; turns: {s.model_turns}"
            if s.reprompts:
                line += f"; reprompts: {s.reprompts}"
            if s.join is not None:
                line += f"; join {s.join}"
            if s.status != "ok" and s.reason is not None:
                line += f"; {s.reason}"
            lines.append(line)
        if self.halted_at is not None:
            lines.append(f"Stopped at step: '{self.halted_at}'.")
        if self.reason is not None:
            lines.append(f"Stopped because: {self.reason}.")
        return "\n".join(lines)
