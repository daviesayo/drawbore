"""Composable instructions: ``instructions`` may be a string or a sequence of
fragments composed deterministically at decoration time.

The normalisation lands in the frozen ``AgentSpec`` as a single ``str | None``,
so everything downstream (``build.py``, the manifest round-trip) is unchanged.
"""

from pydantic import BaseModel

from drawbore import agent
from drawbore.agent.decorator import _compose_instructions
from drawbore.llm.build import build_model_request


class _In(BaseModel):
    x: int


class _Out(BaseModel):
    y: int


def _spec(instructions):
    @agent(name="a", input=_In, output=_Out, instructions=instructions)
    async def fn(value: _In) -> _Out:
        return _Out(y=value.x)

    return fn.spec


def test_plain_str_unchanged():
    assert _spec("hello").instructions == "hello"


def test_none_unchanged():
    assert _spec(None).instructions is None


def test_sequence_joined_in_order():
    assert _spec(["first", "second", "third"]).instructions == "first\n\nsecond\n\nthird"


def test_tuple_is_accepted():
    assert _spec(("p", "q")).instructions == "p\n\nq"


def test_blank_fragments_dropped():
    assert _spec(["a", "  ", "", "b"]).instructions == "a\n\nb"


def test_fragments_are_stripped():
    assert _spec([" a ", "\tb\n"]).instructions == "a\n\nb"


def test_all_blank_sequence_normalises_to_none():
    assert _spec(["", "   "]).instructions is None
    assert _spec([]).instructions is None


def test_compose_helper_directly():
    assert _compose_instructions(None) is None
    assert _compose_instructions("x") == "x"
    assert _compose_instructions(["a", "b"]) == "a\n\nb"
    assert _compose_instructions([]) is None


def test_build_request_identical_for_str_and_fragments():
    joined = _spec("alpha\n\nbeta")
    fragments = _spec(["alpha", "beta"])
    req_joined = build_model_request(joined, _In(x=1), model_chain=("gpt-4o",))
    req_fragments = build_model_request(fragments, _In(x=1), model_chain=("gpt-4o",))
    assert req_joined.system == req_fragments.system
