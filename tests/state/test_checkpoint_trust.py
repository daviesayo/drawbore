"""Tests for per-step taint persistence on CheckpointStore.

Covers:
- TrustLabel round-trips through InMemoryCheckpointStore (UNTRUSTED and TRUSTED).
- Default fail-closed behaviour: a store with no record_trust call returns UNTRUSTED.
- ABC default path: a minimal subclass that does NOT override record_trust/trust_of
  silently discards writes and always returns UNTRUSTED.
- Per-key isolation: recording trust for one (run_id, step) does not affect others.
"""

from pydantic import BaseModel

from drawbore.state import CheckpointStore, InMemoryCheckpointStore
from drawbore.tools import TrustLabel


class Out(BaseModel):
    v: int


# ---------------------------------------------------------------------------
# Round-trip tests (InMemoryCheckpointStore)
# ---------------------------------------------------------------------------

def test_trust_round_trips_untrusted():
    store = InMemoryCheckpointStore()
    store.step_succeeded("r", 0, Out(v=1))
    store.record_trust("r", 0, TrustLabel.UNTRUSTED)
    assert store.trust_of("r", 0) is TrustLabel.UNTRUSTED


def test_trust_round_trips_trusted():
    # Regression guard: wrong key or wrong default would make this return UNTRUSTED.
    store = InMemoryCheckpointStore()
    store.step_succeeded("r", 0, Out(v=1))
    store.record_trust("r", 0, TrustLabel.TRUSTED)
    assert store.trust_of("r", 0) is TrustLabel.TRUSTED


def test_trust_of_defaults_fail_closed_when_unrecorded():
    store = InMemoryCheckpointStore()
    store.step_succeeded("r", 0, Out(v=1))  # no record_trust call — tests the default
    assert store.trust_of("r", 0) is TrustLabel.UNTRUSTED  # fail closed, never TRUSTED


# ---------------------------------------------------------------------------
# ABC default path (minimal subclass, no record_trust/trust_of override)
# ---------------------------------------------------------------------------

class _MinimalStore(CheckpointStore):
    """Implements only the pre-existing abstract methods; relies on ABC defaults."""

    def step_started(self, run_id: str, step: int) -> None:
        return None

    def step_succeeded(self, run_id: str, step: int, output: BaseModel) -> None:
        return None

    def is_completed(self, run_id: str, step: int) -> bool:
        return False

    def output_of(self, run_id: str, step: int) -> BaseModel:
        raise KeyError(f"step {step} of run {run_id!r} not stored")

    def step_skipped(self, run_id: str, step: int) -> None:
        return None

    def is_skipped(self, run_id: str, step: int) -> bool:
        return False

    def record_fingerprint(self, run_id: str, fingerprint: str) -> None:
        return None

    def fingerprint_matches(self, run_id: str, fingerprint: str) -> bool:
        return True


def test_abc_default_record_trust_is_silent_noop():
    store = _MinimalStore()
    # Must not raise; return value is unspecified (None by convention).
    store.record_trust("r", 0, TrustLabel.TRUSTED)


def test_abc_default_trust_of_returns_untrusted_before_and_after_record():
    store = _MinimalStore()
    assert store.trust_of("r", 0) is TrustLabel.UNTRUSTED
    store.record_trust("r", 0, TrustLabel.TRUSTED)  # silent discard
    assert store.trust_of("r", 0) is TrustLabel.UNTRUSTED  # still fail-closed


# ---------------------------------------------------------------------------
# Per-key isolation (InMemoryCheckpointStore)
# ---------------------------------------------------------------------------

def test_trust_is_isolated_to_recorded_key():
    store = InMemoryCheckpointStore()
    store.record_trust("r", 0, TrustLabel.TRUSTED)
    # Different step index: must remain UNTRUSTED.
    assert store.trust_of("r", 1) is TrustLabel.UNTRUSTED
    # Different run_id: must remain UNTRUSTED.
    assert store.trust_of("r2", 0) is TrustLabel.UNTRUSTED
