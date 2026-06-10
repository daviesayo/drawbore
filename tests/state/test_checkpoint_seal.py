"""Seal persistence on the checkpoint store, and the fail-closed default
for stores that do not persist seals."""

from drawbore.state import CheckpointStore, InMemoryCheckpointStore
from drawbore.state.step_seal import StepSeal


def _seal(**overrides) -> StepSeal:
    base = dict(
        agent="a",
        version="1.0.0",
        risk_tier="low",
        requires_human_approval=False,
        context_access="none",
        tools=(),
        model=None,
        fallback_model=None,
        instructions_fingerprint="none",
        input_schema_fingerprint="sha256:aa",
        output_schema_fingerprint="sha256:bb",
        evidence_policy_fingerprint="none",
    )
    base.update(overrides)
    return StepSeal(**base)


class _MinimalStore(CheckpointStore):
    """A store implementing only the abstract surface (an un-upgraded store)."""

    def __init__(self):
        self._outputs = {}
        self._fingerprints = {}

    def step_started(self, run_id, step):
        pass

    def step_succeeded(self, run_id, step, output):
        self._outputs[(run_id, step)] = output

    def is_completed(self, run_id, step):
        return (run_id, step) in self._outputs

    def output_of(self, run_id, step):
        return self._outputs[(run_id, step)]

    def step_skipped(self, run_id, step):
        pass

    def is_skipped(self, run_id, step):
        return False

    def record_fingerprint(self, run_id, fingerprint):
        self._fingerprints[run_id] = fingerprint

    def fingerprint_matches(self, run_id, fingerprint):
        stored = self._fingerprints.get(run_id)
        return stored is None or stored == fingerprint


def test_in_memory_round_trips_seal():
    store = InMemoryCheckpointStore()
    seal = _seal()
    store.record_seal("r1", 0, seal)
    assert store.seal_of("r1", 0) == seal


def test_in_memory_seal_missing_is_none():
    store = InMemoryCheckpointStore()
    assert store.seal_of("r1", 0) is None


def test_unupgraded_store_returns_none_for_seal():
    store = _MinimalStore()
    store.record_seal("r1", 0, _seal())  # default no-op
    assert store.seal_of("r1", 0) is None


def test_record_fingerprint_overwrites():
    # The pipeline resets a stale fingerprint when a topology change is found
    # with zero prior progress; the in-memory store must overwrite, not keep
    # the first value.
    store = InMemoryCheckpointStore()
    store.record_fingerprint("r1", "sha256:old")
    store.record_fingerprint("r1", "sha256:new")
    assert store.fingerprint_matches("r1", "sha256:new")
    assert not store.fingerprint_matches("r1", "sha256:old")
