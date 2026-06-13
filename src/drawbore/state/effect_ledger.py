"""Durable effect ledger — exactly-once resume for effectful tool calls.

Records each successfully-executed effectful tool call so that, on crash+resume
of a pipeline step, already-fired effects are replayed from the ledger rather
than re-fired. When a resumed execution diverges from the recorded sequence the
run halts fail-closed.

``EffectLedger`` is the interface; ``InMemoryEffectLedger`` is the in-process
default (mirrors the ``CheckpointStore``/``EvidenceStore`` pattern). Durable
backends are managed-service.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Canonical order-stable hash — used for effect matching, NOT for proxy logs.
# ---------------------------------------------------------------------------


def ledger_args_hash(args: Any) -> str:
    """Canonical, order-stable SHA-256 over tool args, for the effect ledger.

    Deliberately NOT the proxy's repr()-based payload_hash (which is dict-order
    and int/float unstable and would make every model-driven resume a false
    divergence).
    """
    canonical = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# EffectStatus
# ---------------------------------------------------------------------------


class EffectStatus(str, Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"


# ---------------------------------------------------------------------------
# EffectEntry — one recorded effectful call
# ---------------------------------------------------------------------------


class EffectEntry(BaseModel):
    """A single effectful tool-call record in the ledger.

    ``status=PENDING`` means the handler fired but durable confirmation was
    not written; ``status=SUCCEEDED`` means the call completed and its output
    is stored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    step: int
    position: int           # 0-based ordinal of this effectful call within the step
    tool_ref: str
    input_hash: str
    idempotency_key: str    # hex sha256 of [run_id, str(step), str(position), tool_ref, input_hash]
    status: EffectStatus
    output: Any | None      # None while PENDING


# ---------------------------------------------------------------------------
# EffectLedger ABC
# ---------------------------------------------------------------------------


class EffectLedger(ABC):
    """Durable record of effectful tool calls within a pipeline step.

    All methods are **synchronous**. A durable backend that requires async I/O
    must bridge it internally (e.g. via a thread executor); callers are always
    synchronous here (matching ``CheckpointStore`` and ``EvidenceStore``).

    **Inter-store note (benign):** the proxy's ``record_succeeded`` (on the
    effect ledger) and the pipeline's ``checkpoints.step_succeeded`` are separate
    stores with no cross-store atomic write. If the process crashes after the
    checkpoint ``step_succeeded`` but before the ledger's ``record_succeeded``,
    the step is marked completed (so resume short-circuits it) but a ``PENDING``
    entry may remain in the ledger. This entry is never re-resolved — the step is
    never re-run — so no double-fire occurs. A ``PENDING`` entry under a completed
    step can look anomalous to an auditor; resolving such orphan entries is part of
    the deferred operator-resolution API.
    """

    @abstractmethod
    def record_pending(self, entry: EffectEntry) -> None:
        """Write ``entry`` (with ``status=PENDING``) before the handler fires.

        Must persist durably *before* the handler executes — a crash between
        this write and the handler means a ``PENDING`` entry is found on resume,
        which correctly halts ``effect_unresolved``.
        """

    @abstractmethod
    def record_succeeded(
        self, run_id: str, step: int, position: int, output: Any
    ) -> None:
        """Rewrite the entry at ``(run_id, step, position)`` to ``SUCCEEDED``
        with the given ``output``. Called immediately after the handler returns.
        A failure here must propagate as ``effect_ledger_error`` — the result is
        never returned if it cannot be durably recorded.
        """

    @abstractmethod
    def entry_at(
        self, run_id: str, step: int, position: int
    ) -> EffectEntry | None:
        """Return the entry at ``(run_id, step, position)``, or ``None``."""

    @abstractmethod
    def recorded_count(self, run_id: str, step: int) -> int:
        """Total number of recorded entries for ``(run_id, step)``.

        Used by the step-end orphan check: if the resumed run's cursor is below
        this count, recorded effects were skipped — halt ``effect_divergence``.
        """

    @abstractmethod
    def entries_from(
        self, run_id: str, step: int, start_pos: int
    ) -> list[EffectEntry]:
        """All recorded entries for ``(run_id, step)`` with
        ``position >= start_pos``, sorted by position ascending.

        Used to enumerate orphaned (unconsumed) recorded entries for the
        legible divergence halt message.
        """


# ---------------------------------------------------------------------------
# InMemoryEffectLedger
# ---------------------------------------------------------------------------


class InMemoryEffectLedger(EffectLedger):
    """In-process effect ledger.

    Backed by a dict keyed on ``(run_id, step, position)``. Suitable for
    local testing and single-process runs; does not survive process restart
    (use a durable backend for real crash+resume).
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, int, int], EffectEntry] = {}

    def record_pending(self, entry: EffectEntry) -> None:
        self._entries[(entry.run_id, entry.step, entry.position)] = entry

    def record_succeeded(
        self, run_id: str, step: int, position: int, output: Any
    ) -> None:
        existing = self._entries.get((run_id, step, position))
        if existing is None:
            raise EffectLedgerWriteError(
                f"no pending effect entry at ({run_id!r}, step={step}, pos={position}):"
                " record_pending must be called before record_succeeded"
            )
        self._entries[(run_id, step, position)] = existing.model_copy(
            update={"status": EffectStatus.SUCCEEDED, "output": output}
        )

    def entry_at(
        self, run_id: str, step: int, position: int
    ) -> EffectEntry | None:
        return self._entries.get((run_id, step, position))

    def recorded_count(self, run_id: str, step: int) -> int:
        return sum(1 for (r, s, _) in self._entries if r == run_id and s == step)

    def entries_from(
        self, run_id: str, step: int, start_pos: int
    ) -> list[EffectEntry]:
        return sorted(
            (
                entry
                for (r, s, pos), entry in self._entries.items()
                if r == run_id and s == step and pos >= start_pos
            ),
            key=lambda e: e.position,
        )


# ---------------------------------------------------------------------------
# Halt exceptions — self-declaring via halt_reason class attribute so that
# drawbore.errors does not need to import this module (avoids a cycle).
# ---------------------------------------------------------------------------


class EffectDivergenceError(Exception):
    """Raised when a resumed tool call does not match the recorded effect sequence,
    or when recorded effects remain unconsumed at step end."""

    halt_reason = "effect_divergence"


class EffectUnresolvedError(Exception):
    """Raised when a resume encounters a PENDING effect entry — the framework
    cannot prove the original call succeeded, so it halts fail-closed."""

    halt_reason = "effect_unresolved"


class EffectLedgerWriteError(Exception):
    """Raised when a durable effect-ledger write fails. The result is not returned
    rather than risk an unrecorded effect."""

    halt_reason = "effect_ledger_error"
