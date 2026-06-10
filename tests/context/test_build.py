from pydantic import BaseModel
from drawbore.context import build_input


class Out(BaseModel):
    doubled: int
    label: str


class _From:
    """Minimal binding stand-in (duck-typed: .agent, .field)."""
    def __init__(self, agent, field=None):
        self.agent = agent
        self.field = field


def test_first_step_no_bindings_returns_initial():
    initial = Out(doubled=1, label="x")
    assert build_input({}, {}, initial=initial, predecessor=None) is initial


def test_linear_step_no_bindings_returns_predecessor_dump():
    prev = Out(doubled=10, label="d10")
    payload = build_input({}, {"a": prev}, predecessor="a")
    assert payload == {"doubled": 10, "label": "d10"}


def test_fan_in_builds_dict_from_bindings():
    a = Out(doubled=10, label="la")
    b = Out(doubled=99, label="lb")
    inputs = {"doubled": _From("a", "doubled"), "label": _From("b", "label")}
    payload = build_input(inputs, {"a": a, "b": b})
    assert payload == {"doubled": 10, "label": "lb"}


def test_whole_output_binding_returns_dump():
    a = Out(doubled=10, label="la")
    inputs = {"whole": _From("a", None)}
    payload = build_input(inputs, {"a": a})
    assert payload == {"whole": {"doubled": 10, "label": "la"}}


def test_first_step_with_no_initial_returns_none():
    # No bindings and no initial (predecessor None) -> None is the constructed payload.
    assert build_input({}, {}, initial=None, predecessor=None) is None
