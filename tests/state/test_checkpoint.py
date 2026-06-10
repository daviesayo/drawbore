import pytest
from pydantic import BaseModel
from drawbore.state import CheckpointStore, InMemoryCheckpointStore


class Out(BaseModel):
    v: int


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        CheckpointStore()


def test_started_does_not_mark_completed():
    s = InMemoryCheckpointStore()
    s.step_started("run-1", 0)
    assert s.is_completed("run-1", 0) is False


def test_succeeded_marks_completed_and_stores_output():
    s = InMemoryCheckpointStore()
    out = Out(v=42)
    s.step_succeeded("run-1", 0, out)
    assert s.is_completed("run-1", 0) is True
    assert s.output_of("run-1", 0) == out


def test_runs_are_isolated_by_run_id():
    s = InMemoryCheckpointStore()
    s.step_succeeded("run-1", 0, Out(v=1))
    assert s.is_completed("run-2", 0) is False
