"""Append-only sink for admission verdicts.

Mirrors the audit sink pattern: an ABC write seam plus an in-process default.
Admission records are a separate surface from run audit records -- an admission
may precede many runs or none (a rejection has no run at all) -- so they are
never folded into the run audit model; the two join on the manifest fingerprint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .verdict import RatchetVerdict


class RatchetSink(ABC):
    """Write-only destination for admission verdicts (append-only contract)."""

    @abstractmethod
    def write(self, verdict: RatchetVerdict) -> None:
        """Append ``verdict``. Implementations MUST NOT mutate or drop prior
        records."""


class InMemoryRatchetSink(RatchetSink):
    """In-process append-only sink; ``verdicts`` returns a copy."""

    def __init__(self) -> None:
        self._verdicts: list[RatchetVerdict] = []

    def write(self, verdict: RatchetVerdict) -> None:
        self._verdicts.append(verdict)

    @property
    def verdicts(self) -> list[RatchetVerdict]:
        return list(self._verdicts)
