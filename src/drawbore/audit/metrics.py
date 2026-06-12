"""Quantitative run metrics — the cost/performance view of a run.

A separate concern from the legible ``AuditRecord`` (dispositions, hashes, tool-call
strings): ``RunMetrics`` is the numbers a cost or performance analysis needs — per-step
wall-clock duration, per-step token usage and provider-reported cost (each ``None`` when
the provider does not report it, never fabricated), and the tool-proxy call log (tool,
operation, duration, disposition). It is attached to ``RunResult.metrics`` on every run
and correlates with the audit trace and spans by ``run_id`` and step index. ``to_dict``
renders the whole object to JSON-safe primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotation-only: drawbore.llm transitively imports litellm, so the audit
    # layer never imports it at runtime (mirrors records.py / ModelAudit).
    from drawbore.llm import TokenUsage


@dataclass(frozen=True)
class ToolCallMetric:
    """One tool call as the proxy measured it: which tool, the operation, the
    wall-clock duration in seconds, and the disposition (``"ok"``, ``"error"``, or a
    ``"denied:*"`` reason)."""

    tool: str
    operation: str
    duration_seconds: float
    result: str

    @classmethod
    def from_log_entry(cls, entry: dict) -> "ToolCallMetric":
        """Build from a tool-proxy log entry (the proxy records one dict per call)."""
        return cls(
            tool=entry["tool"],
            operation=entry["operation"],
            duration_seconds=float(entry["duration"]),
            result=entry["result"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "operation": self.operation,
            "duration_seconds": self.duration_seconds,
            "result": self.result,
        }


@dataclass(frozen=True)
class StepMetric:
    """One recorded step's quantitative summary. ``duration_seconds`` is the step's
    wall-clock time; ``tokens`` and ``cost`` carry the model call's token usage and
    provider-reported cost, each ``None`` for a deterministic step or when the provider
    did not report it."""

    index: int
    agent: str
    duration_seconds: float | None
    tokens: "TokenUsage | None" = None
    cost: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "agent": self.agent,
            "duration_seconds": self.duration_seconds,
            "tokens": self.tokens.to_dict() if self.tokens is not None else None,
            "cost": self.cost,
        }


@dataclass(frozen=True)
class RunMetrics:
    """The quantitative metrics for one run: per-step summaries and the tool-proxy
    call log. Present on every ``RunResult``; correlate to the audit trace by
    ``run_id`` and step index."""

    run_id: str
    steps: tuple[StepMetric, ...]
    tool_calls: tuple[ToolCallMetric, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "steps": [s.to_dict() for s in self.steps],
            "tool_calls": [c.to_dict() for c in self.tool_calls],
        }
