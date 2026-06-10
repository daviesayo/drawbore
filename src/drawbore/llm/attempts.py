"""Provider-attempt audit shape.

Owned by ``drawbore.llm``. ``drawbore.audit.StepAuditRecord`` carries a
``ModelAudit | None`` (string-annotated; no runtime import of ``llm``) and renders it
via this ``legible()``. Secret hygiene: NEVER put API keys, headers,
env values, raw prompts, or raw outputs in these records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelAttemptAudit:
    """One provider attempt's audited outcome."""

    index: int
    declared_ref: str
    source: Literal["profile", "direct"]
    provider: str | None
    model: str
    request_model: str
    outcome: Literal["success", "fallback", "halted"]
    reason: str | None = None


@dataclass(frozen=True)
class ModelAudit:
    """The model-attempt summary for one step (additive; recorded on
    ``StepAuditRecord.model``). ``loop_fallback_phase`` distinguishes a one-shot from a
    loop, and for a loop whether fallback happened before tools or halted after."""

    declared_refs: tuple[str, ...]
    selected_provider: str | None
    selected_model: str | None
    attempts: tuple[ModelAttemptAudit, ...]
    loop_fallback_phase: Literal["not_loop", "before_tools", "after_tools_halted"] = "not_loop"

    def legible(self) -> str:
        """One-line model summary for the audit trace."""
        refs = ", ".join(self.declared_refs)
        parts = [f"model {refs}"]
        for a in self.attempts:
            line = f"{a.request_model} -> {a.outcome}"
            if a.reason:
                line += f" ({a.reason})"
            parts.append(line)
        if self.selected_model is not None:
            # Prefer a provider/model form; if the model is already provider-qualified
            # (contains '/') use it as-is to avoid a double prefix.
            if "/" in self.selected_model or not self.selected_provider:
                selected = self.selected_model
            else:
                selected = f"{self.selected_provider}/{self.selected_model}"
            parts.append(f"selected {selected}")
        if self.loop_fallback_phase != "not_loop":
            parts.append(f"loop fallback: {self.loop_fallback_phase}")
        return "; ".join(parts)
