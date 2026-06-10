from drawbore.audit import AuditRecorder, StepAuditRecord


def test_step_record_has_model_turns_defaulting_to_zero():
    s = StepAuditRecord(
        index=0, agent="a", version="1.0.0", agent_id=None, status="ok",
        input_hash="i" * 16, output_hash="o" * 16, tool_calls=(),
    )
    assert s.model_turns == 0


def test_recorder_records_model_turns_on_a_successful_step():
    rec = AuditRecorder(run_id="r1", pipeline="p", version="0.0.0", tenant_id=None)
    rec.record_step(
        index=0, agent="screen", version="1.0.0", agent_id=None,
        input_hash="i" * 16, output_hash="o" * 16, tool_calls=("t (invoke) -> ok",),
        model_turns=3,
    )
    record = rec.build(status="completed", reason=None, halted_at=None, escalations=0)
    assert record.steps == 1
    assert record.step_records[0].model_turns == 3


def test_recorder_records_a_failed_step_not_counted_in_steps():
    rec = AuditRecorder(run_id="r1", pipeline="p", version="0.0.0", tenant_id=None)
    rec.record_step(
        index=0, agent="ok-step", version="1.0.0", agent_id=None,
        input_hash="i" * 16, output_hash="o" * 16, tool_calls=(),
    )
    rec.record_failed_step(
        index=1, agent="loop", version="1.0.0", agent_id=None,
        input_hash="i" * 16, tool_calls=("risk (invoke) -> denied:scope",),
        reason="denied:scope: tool 'risk' ...", model_turns=2,
    )
    record = rec.build(status="halted", reason="denied:scope: tool 'risk' ...",
                       halted_at="loop", escalations=0)
    # the failed step is NOT counted in `steps` (successful only) ...
    assert record.steps == 1
    # ... but it IS in the trail, with its tool calls + reason + turns.
    failed = record.step_records[1]
    assert failed.status == "failed"
    assert failed.tool_calls == ("risk (invoke) -> denied:scope",)
    assert "denied:scope" in failed.reason
    assert failed.model_turns == 2
    text = record.legible()
    assert "risk (invoke) -> denied:scope" in text
    assert "denied:scope" in text
    assert "turns: 2" in text       # model turns are legible too, not only on the record
