from drawbore.audit import AuditRecord, AuditSink, InMemoryAuditSink


def _rec(run_id):
    return AuditRecord(
        run_id=run_id, pipeline="p", version="0.0.0", tenant_id=None,
        status="completed", steps=0, escalations=0, schema_violations=0,
        reason=None, step_records=(),
    )


def test_in_memory_sink_is_append_only_and_queryable():
    sink = InMemoryAuditSink()
    sink.write(_rec("r1"))
    sink.write(_rec("r2"))
    assert [r.run_id for r in sink.records] == ["r1", "r2"]
    assert sink.by_run_id("r2").run_id == "r2"
    assert sink.by_run_id("nope") is None


def test_in_memory_sink_records_view_is_a_copy():
    # Append-only: callers cannot mutate the sink's history through the view.
    sink = InMemoryAuditSink()
    sink.write(_rec("r1"))
    view = sink.records
    view.clear()
    assert len(sink.records) == 1


def test_abc_cannot_be_instantiated():
    import pytest
    with pytest.raises(TypeError):
        AuditSink()
