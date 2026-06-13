from drawbore.audit import AuditRecorder


def _recorder():
    return AuditRecorder(run_id="r-1", pipeline="p", version="1.0.0", tenant_id=None)


def test_record_step_carries_approval_fields():
    rec = _recorder()
    rec.record_step(
        index=0, agent="scorer", version="0.0.0", agent_id=None,
        input_hash=None, output_hash="sha256:" + "ab" * 32, tool_calls=(),
        human_decision="amended", reviewer_id="rev-1",
        amendment_original_hash="sha256:" + "11" * 32,
        amendment_applied_hash="sha256:" + "22" * 32,
    )
    record = rec.build(status="completed", reason=None, halted_at=None, escalations=0)
    step = record.step_records[0]
    assert step.human_decision == "amended"
    assert step.reviewer_id == "rev-1"
    text = record.legible()
    assert "amended by reviewer 'rev-1'" in text
    assert ("sha256:" + "11" * 32) in text and ("sha256:" + "22" * 32) in text


def test_approval_fields_default_to_none_and_render_nothing():
    rec = _recorder()
    rec.record_step(
        index=0, agent="scorer", version="0.0.0", agent_id=None,
        input_hash=None, output_hash=None, tool_calls=(),
    )
    record = rec.build(status="completed", reason=None, halted_at=None, escalations=0)
    assert record.step_records[0].human_decision is None
    assert "reviewer" not in record.legible()
