from drawbore.state import RunState


def test_run_state_defaults():
    s = RunState(run_id="abc")
    assert s.run_id == "abc"
    assert s.step == 0
    assert s.error_count == 0


def test_run_state_is_mutable_by_orchestrator():
    s = RunState(run_id="abc")
    s.step = 2
    s.error_count += 1
    assert s.step == 2
    assert s.error_count == 1
