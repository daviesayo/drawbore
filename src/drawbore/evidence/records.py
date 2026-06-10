"""Evidence records — the immutable, audit-visible facts.

``EvidenceHandle`` is a model-visible *reference* to compressed evidence; it is
NOT a capability (the proxy is authoritative). ``EvidenceDecision``
is the immutable description of what the layer did, for audit/debug/tests. Both
are frozen dataclasses; ``EvidencePolicy`` (the
user-facing, validated config) is a Pydantic model in ``policy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EvidenceHandle:
    """A reference to a stored original + its compressed view. Data, not authority:
    a handle appearing in model input/audit does not by itself permit retrieval
    ``handle_id`` is deterministic."""

    handle_id: str
    run_id: str
    step: int
    source_agent: str
    content_type: str
    original_hash: str
    compressed_hash: str
    original_tokens: int
    compressed_tokens: int
    transform: str


@dataclass(frozen=True)
class EvidenceDecision:
    """What the evidence layer did for one step's payload — for audit."""

    decision: Literal["compressed", "passthrough", "denied", "failed"]
    reason: str
    policy: str
    transform: str | None
    original_tokens: int
    compressed_tokens: int | None
    handle_id: str | None
    warnings: tuple[str, ...] = ()

    def legible(self) -> str:
        """One-line, non-engineer-readable summary for the audit trail."""
        parts = [f"evidence {self.decision} ({self.reason})"]
        if self.transform is not None:
            parts.append(f"via {self.transform}")
        if self.compressed_tokens is not None:
            parts.append(f"{self.original_tokens}->{self.compressed_tokens} tokens")
        else:
            parts.append(f"{self.original_tokens} tokens")
        if self.handle_id is not None:
            parts.append(f"handle {self.handle_id}")
        line = "; ".join(parts)
        if self.warnings:
            line += f" [warnings: {', '.join(self.warnings)}]"
        return line
