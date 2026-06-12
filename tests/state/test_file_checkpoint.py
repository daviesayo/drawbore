"""Durable, file-backed checkpoint store.

These tests pin the store in isolation: a fresh instance reading a directory
written by a prior instance (a process restart) must restore outputs, skip
marks, the topology fingerprint, per-step seals, and trust labels. Reconstruction
of typed outputs uses the live pipeline's output models supplied via
``bind_models`` — never a class path read from disk.
"""

from __future__ import annotations

from pydantic import BaseModel

from drawbore.state import FileCheckpointStore
from drawbore.state.step_seal import StepSeal
from drawbore.tools.taint import TrustLabel


class Out(BaseModel):
    v: int
    label: str = "x"


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


def test_output_round_trips_across_a_new_instance(tmp_path):
    """A second instance from the same directory restores a typed output."""
    s1 = FileCheckpointStore(tmp_path)
    s1.step_succeeded("r1", 0, Out(v=42, label="hello"))

    s2 = FileCheckpointStore(tmp_path)
    s2.bind_models("r1", {0: Out})
    assert s2.is_completed("r1", 0) is True
    restored = s2.output_of("r1", 0)
    assert isinstance(restored, Out)
    assert restored == Out(v=42, label="hello")


def test_seal_round_trips_across_a_new_instance(tmp_path):
    s1 = FileCheckpointStore(tmp_path)
    seal = _seal(version="2.3.4")
    s1.step_succeeded("r1", 0, Out(v=1))
    s1.record_seal("r1", 0, seal)

    s2 = FileCheckpointStore(tmp_path)
    assert s2.seal_of("r1", 0) == seal


def test_trust_round_trips_across_a_new_instance(tmp_path):
    s1 = FileCheckpointStore(tmp_path)
    s1.step_succeeded("r1", 0, Out(v=1))
    s1.record_trust("r1", 0, TrustLabel.UNTRUSTED)
    s1.step_succeeded("r1", 1, Out(v=2))
    s1.record_trust("r1", 1, TrustLabel.TRUSTED)

    s2 = FileCheckpointStore(tmp_path)
    assert s2.trust_of("r1", 0) is TrustLabel.UNTRUSTED
    assert s2.trust_of("r1", 1) is TrustLabel.TRUSTED


def test_trust_defaults_untrusted_for_unknown_step(tmp_path):
    s = FileCheckpointStore(tmp_path)
    assert s.trust_of("r1", 99) is TrustLabel.UNTRUSTED


def test_seal_missing_is_none(tmp_path):
    s = FileCheckpointStore(tmp_path)
    s.step_succeeded("r1", 0, Out(v=1))
    assert s.seal_of("r1", 0) is None


def test_skip_marks_round_trip(tmp_path):
    s1 = FileCheckpointStore(tmp_path)
    s1.step_skipped("r1", 2)

    s2 = FileCheckpointStore(tmp_path)
    assert s2.is_skipped("r1", 2) is True
    assert s2.is_skipped("r1", 0) is False
    assert s2.is_completed("r1", 2) is False


def test_fingerprint_round_trip_and_overwrite(tmp_path):
    s1 = FileCheckpointStore(tmp_path)
    s1.record_fingerprint("r1", "sha256:old")

    s2 = FileCheckpointStore(tmp_path)
    assert s2.fingerprint_matches("r1", "sha256:old")
    assert not s2.fingerprint_matches("r1", "sha256:new")
    # Unknown run: no fingerprint stored yet -> always matches.
    assert s2.fingerprint_matches("unknown", "sha256:anything")
    s2.record_fingerprint("r1", "sha256:new")

    s3 = FileCheckpointStore(tmp_path)
    assert s3.fingerprint_matches("r1", "sha256:new")
    assert not s3.fingerprint_matches("r1", "sha256:old")


def test_runs_are_isolated_by_run_id(tmp_path):
    s = FileCheckpointStore(tmp_path)
    s.step_succeeded("r1", 0, Out(v=1))
    assert s.is_completed("r2", 0) is False


def test_run_ids_with_unsafe_characters_are_isolated(tmp_path):
    """Two run ids that differ only by path-unsafe characters must not collide."""
    s = FileCheckpointStore(tmp_path)
    s.step_succeeded("a/../b", 0, Out(v=1, label="A"))
    s.step_succeeded("a__b", 0, Out(v=2, label="B"))
    s.bind_models("a/../b", {0: Out})
    s.bind_models("a__b", {0: Out})
    assert s.output_of("a/../b", 0) == Out(v=1, label="A")
    assert s.output_of("a__b", 0) == Out(v=2, label="B")


def test_output_of_without_bound_model_fails_closed(tmp_path):
    """A durable store cannot fabricate a type: reading an output before its
    live model is bound must raise, never guess or import a persisted path."""
    s1 = FileCheckpointStore(tmp_path)
    s1.step_succeeded("r1", 0, Out(v=1))
    s2 = FileCheckpointStore(tmp_path)
    try:
        s2.output_of("r1", 0)
    except Exception:
        return
    raise AssertionError("output_of must fail closed when no model is bound")


def test_torn_temp_write_does_not_corrupt_a_prior_good_checkpoint(tmp_path):
    """A crash mid-write leaves a stray temp file; the prior good checkpoint
    must still load on a fresh instance, and the temp file must be ignored."""
    import os

    s1 = FileCheckpointStore(tmp_path)
    s1.step_succeeded("r1", 0, Out(v=7, label="good"))
    s1.record_seal("r1", 0, _seal())

    # Simulate a crash partway through a later atomic write: a half-written temp
    # file that was never renamed into place. The store must never read it.
    run_dir = s1._run_dir("r1")
    torn = run_dir / "step-00001.json.tmp-deadbeef"
    torn.write_text("{ this is not valid json")
    assert torn.exists()

    s2 = FileCheckpointStore(tmp_path)
    s2.bind_models("r1", {0: Out})
    assert s2.is_completed("r1", 0) is True
    assert s2.output_of("r1", 0) == Out(v=7, label="good")
    assert s2.seal_of("r1", 0) == _seal()
    # Step 1 was never committed.
    assert s2.is_completed("r1", 1) is False


def test_step_succeeded_write_is_atomic_no_partial_file(tmp_path):
    """After a successful write there is exactly one committed step file and no
    leftover temp file."""
    s = FileCheckpointStore(tmp_path)
    s.step_succeeded("r1", 0, Out(v=1))
    run_dir = s._run_dir("r1")
    files = sorted(p.name for p in run_dir.iterdir())
    assert "step-00000.json" in files
    assert not any(".tmp" in f for f in files)
