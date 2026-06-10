"""Idempotent checkpoint store for pipeline resume.

Each step is checkpointed before execution ("safe to retry this step") and after
success ("safe to skip this step on resume"). A run that partially failed resumes
from the last successful checkpoint rather than from the beginning.

``CheckpointStore`` is the interface; ``InMemoryCheckpointStore`` is the MIT-core
default. Durable stores (e.g. the managed service's Postgres) implement the same
interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from drawbore.tools.taint import TrustLabel


class CheckpointStore(ABC):
    @abstractmethod
    def step_started(self, run_id: str, step: int) -> None:
        """Record that ``step`` is about to execute (safe to retry it)."""

    @abstractmethod
    def step_succeeded(self, run_id: str, step: int, output: BaseModel) -> None:
        """Record ``step``'s successful output (safe to skip it on resume)."""

    @abstractmethod
    def is_completed(self, run_id: str, step: int) -> bool:
        """True if ``step`` already succeeded in a prior attempt of this run."""

    @abstractmethod
    def output_of(self, run_id: str, step: int) -> BaseModel:
        """Return the stored output of a completed ``step``."""

    @abstractmethod
    def step_skipped(self, run_id: str, step: int) -> None:
        """Record that ``step`` was skipped (branch not taken / cascade)."""

    @abstractmethod
    def is_skipped(self, run_id: str, step: int) -> bool:
        """True if ``step`` was recorded skipped in a prior attempt of this run."""

    @abstractmethod
    def record_fingerprint(self, run_id: str, fingerprint: str) -> None:
        """Persist the run's topology fingerprint on first use."""

    @abstractmethod
    def fingerprint_matches(self, run_id: str, fingerprint: str) -> bool:
        """True if no fingerprint is stored yet, or the stored one matches."""

    def record_trust(self, run_id: str, step: int, trust: TrustLabel) -> None:
        """Persist ``step``'s output trust label. Default no-op: an un-upgraded store
        simply doesn't persist trust (and ``trust_of`` then fails closed).

        Durable stores MUST override both ``record_trust`` and ``trust_of`` together.
        If they don't, resumed runs transparently over-restrict (every restored step is
        treated as UNTRUSTED) — safe but possibly halting pipelines that used to pass.
        """
        return None

    def trust_of(self, run_id: str, step: int) -> TrustLabel:
        """Return the persisted trust of a completed ``step``. Default UNTRUSTED --
        a store that doesn't persist trust over-restricts on resume, never under.

        Durable stores MUST override both ``record_trust`` and ``trust_of`` together.
        If they don't, resumed runs transparently over-restrict (every restored step is
        treated as UNTRUSTED) — safe but possibly halting pipelines that used to pass.
        """
        return TrustLabel.UNTRUSTED


class InMemoryCheckpointStore(CheckpointStore):
    def __init__(self) -> None:
        self._outputs: dict[tuple[str, int], BaseModel] = {}
        self._skipped: set[tuple[str, int]] = set()
        self._fingerprints: dict[str, str] = {}
        self._trust: dict[tuple[str, int], TrustLabel] = {}

    def step_started(self, run_id: str, step: int) -> None:
        # No-op for the in-memory store: a process crash wipes it, so there is
        # nothing useful to persist about a step that merely *started*. Durable
        # backends (e.g. Postgres) override this to record in-progress steps for
        # crash detection. Resume only needs `is_completed` / `output_of`.
        return None

    def step_succeeded(self, run_id: str, step: int, output: BaseModel) -> None:
        self._outputs[(run_id, step)] = output

    def is_completed(self, run_id: str, step: int) -> bool:
        return (run_id, step) in self._outputs

    def output_of(self, run_id: str, step: int) -> BaseModel:
        try:
            return self._outputs[(run_id, step)]
        except KeyError:
            raise KeyError(f"step {step} of run {run_id!r} has no checkpointed output") from None

    def step_skipped(self, run_id: str, step: int) -> None:
        self._skipped.add((run_id, step))

    def is_skipped(self, run_id: str, step: int) -> bool:
        return (run_id, step) in self._skipped

    def record_fingerprint(self, run_id: str, fingerprint: str) -> None:
        self._fingerprints.setdefault(run_id, fingerprint)

    def fingerprint_matches(self, run_id: str, fingerprint: str) -> bool:
        stored = self._fingerprints.get(run_id)
        return stored is None or stored == fingerprint

    def record_trust(self, run_id: str, step: int, trust: TrustLabel) -> None:
        self._trust[(run_id, step)] = trust

    def trust_of(self, run_id: str, step: int) -> TrustLabel:
        return self._trust.get((run_id, step), TrustLabel.UNTRUSTED)
