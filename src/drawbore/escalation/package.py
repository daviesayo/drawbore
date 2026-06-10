"""The escalation package — the regulator-legible record of why a run stopped or
was flagged. Legibility-first: a non-engineer must be
able to read it without asking an engineer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable


def _render(value: Any) -> str:
    """Render a value for a non-engineer reader: Pydantic models become their
    field mapping (model_dump) rather than constructor-call syntax."""
    if hasattr(value, "model_dump"):
        return str(value.model_dump())
    return repr(value)


@dataclass(frozen=True)
class EscalationPackage:
    """Everything a human needs to act on a halted or flagged run: which step, what
    it received, what it tried to output, why it was blocked, the run trace, and a
    timestamp.
    """

    step: str
    reason: str
    received: Any
    attempted_output: Any | None
    trace: tuple[str, ...]
    timestamp: str
    agent_id: str | None = None

    def legible(self) -> str:
        lines = [
            f"Run stopped at step '{self.step}'.",
        ]
        if self.agent_id is not None:
            lines.append(f"Agent id: {self.agent_id}.")
        lines += [
            f"Why: {self.reason}.",
            f"Steps completed before this: {', '.join(self.trace) or '(none)'}.",
            f"What the step received: {_render(self.received)}.",
        ]
        if self.attempted_output is not None:
            lines.append(f"What it tried to produce: {_render(self.attempted_output)}.")
        lines.append(f"When: {self.timestamp}.")
        return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_escalation(
    *,
    step: str,
    reason: str,
    received: Any,
    attempted_output: Any | None,
    trace: Iterable[str],
    now: Callable[[], str] | None = None,
    agent_id: str | None = None,
) -> EscalationPackage:
    """Construct an :class:`EscalationPackage`, stamping the current UTC time
    (``now`` is injectable for deterministic tests)."""
    return EscalationPackage(
        step=step,
        reason=reason,
        received=received,
        attempted_output=attempted_output,
        trace=tuple(trace),
        timestamp=(now or _utc_now)(),
        agent_id=agent_id,
    )
