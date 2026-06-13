from drawbore.escalation import ApprovalRequest
from drawbore.state import CheckpointStore, InMemoryCheckpointStore


def _req(request_id="req-1"):
    return ApprovalRequest(
        request_id=request_id, run_id="r-1", step="scorer", question="ok?",
        reason="requires_human_approval", package_legible="stopped",
        proposed_output={"x": 1}, proposed_output_trust="trusted",
    )


def test_abc_defaults_are_no_op_and_none():
    class Bare(CheckpointStore):
        def step_started(self, run_id, step): ...
        def step_succeeded(self, run_id, step, output): ...
        def is_completed(self, run_id, step): return False
        def output_of(self, run_id, step): raise KeyError(step)
        def step_skipped(self, run_id, step): ...
        def is_skipped(self, run_id, step): return False
        def record_fingerprint(self, run_id, fingerprint): ...
        def fingerprint_matches(self, run_id, fingerprint): return True
    bare = Bare()
    bare.record_approval_request("r-1", _req().model_dump(mode="json"))  # no-op default
    assert bare.approval_request_of("r-1") is None   # fail-closed default
    bare.clear_approval_request("r-1")               # no-op default


def test_in_memory_record_lookup_clear_cycle():
    store = InMemoryCheckpointStore()
    assert store.approval_request_of("r-1") is None
    req_dict = _req().model_dump(mode="json")
    store.record_approval_request("r-1", req_dict)
    got = store.approval_request_of("r-1")
    assert got is not None and got["request_id"] == "req-1"
    assert store.approval_request_of("other-run") is None   # keyed by run_id
    store.clear_approval_request("r-1")
    assert store.approval_request_of("r-1") is None
    store.clear_approval_request("r-1")                      # idempotent
