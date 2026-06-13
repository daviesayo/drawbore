"""Tests for CheckpointStore.commit_step — single-call atomic write of
output, trust, and seal.

The ABC provides a concrete default that delegates to three separate calls;
FileCheckpointStore overrides it with a single merged os.replace write.
Both must produce identical readable state.
"""

from __future__ import annotations

from pydantic import BaseModel

from drawbore.state import FileCheckpointStore, InMemoryCheckpointStore
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


# --- ABC default (InMemoryCheckpointStore) -----------------------------------

def test_abc_commit_step_persists_output_trust_and_seal():
    """The ABC default delegates to step_succeeded / record_trust / record_seal."""
    s = InMemoryCheckpointStore()
    output = Out(v=5, label="abc")
    trust = TrustLabel.TRUSTED
    seal = _seal(version="1.0.0")

    s.commit_step("r1", 0, output=output, trust=trust, seal=seal)

    assert s.is_completed("r1", 0) is True
    assert s.output_of("r1", 0) == output
    assert s.trust_of("r1", 0) is trust
    assert s.seal_of("r1", 0) == seal


def test_abc_commit_step_without_seal_leaves_seal_none():
    """When seal is omitted (default None) record_seal is not called."""
    s = InMemoryCheckpointStore()
    output = Out(v=1)
    trust = TrustLabel.UNTRUSTED

    s.commit_step("r1", 0, output=output, trust=trust)

    assert s.is_completed("r1", 0) is True
    assert s.output_of("r1", 0) == output
    assert s.trust_of("r1", 0) is trust
    assert s.seal_of("r1", 0) is None


# --- FileCheckpointStore override --------------------------------------------

def test_file_commit_step_round_trips_output_trust_and_seal(tmp_path):
    """commit_step persists all three fields; a fresh instance reads them back."""
    seal = _seal(version="3.0.0")
    output = Out(v=99, label="commit")
    trust = TrustLabel.TRUSTED

    s = FileCheckpointStore(tmp_path)
    s.bind_models("r1", {0: Out})
    s.commit_step("r1", 0, output=output, trust=trust, seal=seal)

    # Verify from a completely fresh instance (simulates a process restart).
    s2 = FileCheckpointStore(tmp_path)
    s2.bind_models("r1", {0: Out})
    assert s2.is_completed("r1", 0) is True
    assert s2.output_of("r1", 0) == output
    assert s2.trust_of("r1", 0) is trust
    assert s2.seal_of("r1", 0) == seal


def test_file_commit_step_without_seal(tmp_path):
    """commit_step with seal=None persists output and trust but no seal."""
    output = Out(v=1, label="no-seal")
    trust = TrustLabel.UNTRUSTED

    s = FileCheckpointStore(tmp_path)
    s.bind_models("r1", {0: Out})
    s.commit_step("r1", 0, output=output, trust=trust)

    s2 = FileCheckpointStore(tmp_path)
    s2.bind_models("r1", {0: Out})
    assert s2.is_completed("r1", 0) is True
    assert s2.output_of("r1", 0) == output
    assert s2.trust_of("r1", 0) is trust
    assert s2.seal_of("r1", 0) is None


def test_file_commit_step_equivalent_to_three_separate_calls(tmp_path):
    """State from commit_step is byte-equivalent to calling the three
    separate methods in sequence (same on-disk JSON after a round-trip)."""
    seal = _seal(version="2.0.0")
    output = Out(v=42, label="eq")
    trust = TrustLabel.TRUSTED

    # Path A: commit_step
    s_commit = FileCheckpointStore(tmp_path / "commit")
    s_commit.commit_step("r1", 0, output=output, trust=trust, seal=seal)

    # Path B: three separate calls
    s_sep = FileCheckpointStore(tmp_path / "separate")
    s_sep.step_succeeded("r1", 0, output)
    s_sep.record_trust("r1", 0, trust)
    s_sep.record_seal("r1", 0, seal)

    # Compare via fresh readers to rule out any in-memory cache discrepancy.
    r_commit = FileCheckpointStore(tmp_path / "commit")
    r_sep = FileCheckpointStore(tmp_path / "separate")
    r_commit.bind_models("r1", {0: Out})
    r_sep.bind_models("r1", {0: Out})

    assert r_commit.is_completed("r1", 0) == r_sep.is_completed("r1", 0)
    assert r_commit.output_of("r1", 0) == r_sep.output_of("r1", 0)
    assert r_commit.trust_of("r1", 0) == r_sep.trust_of("r1", 0)
    assert r_commit.seal_of("r1", 0) == r_sep.seal_of("r1", 0)


def test_file_commit_step_is_single_write(tmp_path):
    """commit_step must produce exactly one committed step file (one os.replace)
    — not one per field. We verify via the on-disk file count and no stray temps."""
    import os

    seal = _seal()
    output = Out(v=7)
    trust = TrustLabel.TRUSTED

    s = FileCheckpointStore(tmp_path)
    s.commit_step("r1", 0, output=output, trust=trust, seal=seal)

    run_dir = s._run_dir("r1")
    files = [p.name for p in run_dir.iterdir()]
    # Exactly one committed step file; no stray temp files.
    step_files = [f for f in files if f.startswith("step-") and f.endswith(".json")]
    tmp_files = [f for f in files if ".tmp" in f]
    assert len(step_files) == 1
    assert len(tmp_files) == 0
