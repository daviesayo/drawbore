from drawbore.audit.recorder import AuditRecorder


def test_record_skipped_step_appears_and_is_not_counted():
    r = AuditRecorder(run_id="r", pipeline="p", version="1.0.0", tenant_id=None)
    r.record_skipped_step(index=2, agent="eDD", version="1.0.0", agent_id=None,
                          condition="risk.level == 'high' (false)")
    rec = r.build(status="completed", reason=None, halted_at=None, escalations=0)
    assert rec.steps == 0                          # a skip is not a completed step
    skipped = rec.step_records[0]
    assert skipped.status == "skipped"
    assert skipped.condition == "risk.level == 'high' (false)"
    assert "skipped" in rec.legible() and "risk.level == 'high'" in rec.legible()
