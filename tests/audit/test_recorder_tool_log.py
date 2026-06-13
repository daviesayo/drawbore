"""Tests for AuditRecorder.tool_log() — the read accessor for the bound proxy log."""
from drawbore.audit import AuditRecorder


def test_tool_log_returns_bound_proxy_log_list():
    rec = AuditRecorder(run_id="r1", pipeline="p", version="0.0.0", tenant_id=None)
    log: list = []
    rec.bind_tool_log(log)
    # tool_log() returns the same object that was bound
    assert rec.tool_log() is log


def test_tool_log_reflects_mutations_to_the_bound_list():
    rec = AuditRecorder(run_id="r1", pipeline="p", version="0.0.0", tenant_id=None)
    log: list = []
    rec.bind_tool_log(log)
    log.append({"tool": "svc.read", "result": "ok", "step": 0, "scope": "trusted",
                "kind": "custom", "exfil_capable": False,
                "run_id": "r1", "operation": "invoke",
                "input_hash": "abc", "output_hash": "def", "duration": 0.001})
    result = rec.tool_log()
    assert len(result) == 1
    assert result[0]["tool"] == "svc.read"


def test_tool_log_before_bind_returns_none():
    rec = AuditRecorder(run_id="r1", pipeline="p", version="0.0.0", tenant_id=None)
    # Before binding, the internal attribute is None — tool_log() should return None
    # (consistent with _tool_log's initial value)
    assert rec.tool_log() is None
