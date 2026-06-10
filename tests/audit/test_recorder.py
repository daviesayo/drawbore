from drawbore.audit import AuditRecorder


def test_recorder_builds_a_record_with_per_step_detail():
    rec = AuditRecorder(run_id="r1", pipeline="p", version="0.0.0", tenant_id="acme")
    rec.record_step(
        index=0, agent="a0", version="1.0.0", agent_id=None,
        input_hash="i0" + "0" * 14, output_hash="o0" + "0" * 14,
        tool_calls=("greet (invoke) -> ok",),
    )
    rec.record_step(
        index=1, agent="a1", version="2.0.0", agent_id="id-1",
        input_hash="i1" + "0" * 14, output_hash="o1" + "0" * 14,
        tool_calls=(),
    )
    record = rec.build(status="completed", reason=None, halted_at=None, escalations=0)
    assert record.run_id == "r1"
    assert record.tenant_id == "acme"
    assert record.steps == 2
    assert record.escalations == 0
    assert record.schema_violations == 0
    assert [s.agent for s in record.step_records] == ["a0", "a1"]
    assert record.step_records[0].tool_calls == ("greet (invoke) -> ok",)


def test_recorder_counts_schema_violations_from_a_halt_reason():
    rec = AuditRecorder(run_id="r1", pipeline="p", version="0.0.0", tenant_id=None)
    rec.record_step(index=0, agent="a0", version="1.0.0", agent_id=None,
                    input_hash="x" * 16, output_hash="y" * 16, tool_calls=())
    record = rec.build(
        status="halted", reason="schema_violation: output: y: int type required", halted_at="a1", escalations=0,
    )
    assert record.steps == 1            # only the one successful step
    assert record.schema_violations == 1
    assert record.status == "halted"
    assert record.reason == "schema_violation: output: y: int type required"
    assert record.halted_at == "a1"     # the run records where it stopped


def test_recorder_reports_escalations_count():
    rec = AuditRecorder(run_id="r1", pipeline="p", version="0.0.0", tenant_id=None)
    record = rec.build(status="escalated", reason="confidence_below_threshold: ...",
                       halted_at="a0", escalations=2)
    assert record.escalations == 2
    assert record.schema_violations == 0    # not a schema violation
