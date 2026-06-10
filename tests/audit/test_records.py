from drawbore.audit import AuditRecord, StepAuditRecord


def _record(status="completed", steps=2, escalations=0, schema_violations=0,
            reason=None, halted_at=None):
    step_records = tuple(
        StepAuditRecord(
            index=i, agent=f"a{i}", version="1.0.0", agent_id=None,
            status="ok", input_hash="i" * 16, output_hash="o" * 16,
            tool_calls=(), reason=None,
        )
        for i in range(steps)
    )
    return AuditRecord(
        run_id="r1", pipeline="p", version="0.0.0", tenant_id=None,
        status=status, steps=steps, escalations=escalations,
        schema_violations=schema_violations, reason=reason, step_records=step_records,
        halted_at=halted_at,
    )


def test_audit_record_exposes_the_brief_acceptance_shape():
    # result.audit_trace.steps == 4, .escalations == 0, .schema_violations == 0
    rec = _record(steps=4)
    assert rec.steps == 4
    assert rec.escalations == 0
    assert rec.schema_violations == 0


def test_audit_record_is_immutable():
    import dataclasses
    import pytest
    rec = _record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.status = "tampered"


def test_legible_reads_as_a_compliance_trace():
    rec = _record(status="escalated", steps=1, escalations=1,
                  reason="schema_violation: output: y: int type required", halted_at="a1")
    text = rec.legible()
    assert "Run 'r1'" in text
    assert "pipeline 'p'" in text
    assert "escalated" in text
    assert "Steps completed: 1" in text
    assert "Escalations: 1" in text
    assert "Stopped at step: 'a1'" in text   # the compliance reader sees WHERE it halted
    assert "schema_violation: output:" in text


def test_step_record_renders_evidence_in_legible():
    rec = AuditRecord(
        run_id="r1", pipeline="p", version="0.0.0", tenant_id=None,
        status="completed", steps=1, escalations=0, schema_violations=0, reason=None,
        step_records=(StepAuditRecord(
            index=0, agent="screen", version="1.0.0", agent_id=None, status="ok",
            input_hash="i" * 16, output_hash="o" * 16, tool_calls=(),
            evidence="evidence compressed (compressed for model); via json_rows; 4000->900 tokens; handle abc",
        ),),
    )
    assert "json_rows" in rec.legible()
    assert "evidence" in rec.legible()


def test_step_record_is_immutable_and_carries_tool_calls():
    import dataclasses
    import pytest
    s = StepAuditRecord(
        index=0, agent="caller", version="1.0.0", agent_id="id-1",
        status="ok", input_hash="i" * 16, output_hash="o" * 16,
        tool_calls=("mcp://slack/send (invoke) -> ok",), reason=None,
    )
    assert s.tool_calls == ("mcp://slack/send (invoke) -> ok",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.status = "x"
