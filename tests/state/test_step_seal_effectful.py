"""Tests: effectful classification sealed against resume drift."""

from __future__ import annotations

from pydantic import BaseModel

from drawbore.agent.spec import AgentSpec
from drawbore.state.step_seal import StepSeal, diff_seals, seal_for
from drawbore.tools.registry.registry import ToolRegistry


class _In(BaseModel):
    x: str


class _Out(BaseModel):
    y: str


async def _fn(payload: BaseModel) -> BaseModel:
    return _Out(y="ok")


async def _noop(payload: BaseModel) -> BaseModel:
    return _Out(y="noop")


def _registry(*, write_effectful: bool = True, read_effectful: bool = False) -> ToolRegistry:
    r = ToolRegistry()
    r.register_tool("write", _fn, effectful=write_effectful)
    r.register_tool("read", _noop, effectful=read_effectful)
    return r


def _spec(*tools: str) -> AgentSpec:
    return AgentSpec(
        name="agent",
        input=_In,
        output=_Out,
        fn=_fn,
        tools=tools,
        risk_tier="low",
        version="1.0.0",
        model=None,
        fallback_model=None,
        instructions="Do something.",
    )


# ---------------------------------------------------------------------------
# effectful_tools is correctly populated
# ---------------------------------------------------------------------------

def test_effectful_tools_contains_only_effectful_refs():
    """seal_for includes only the effectful tool ref in effectful_tools."""
    r = _registry()  # write=True, read=False
    seal = seal_for(_spec("write", "read"), None, r)
    assert seal.effectful_tools == ("write",)


def test_effectful_tools_empty_when_none_effectful():
    r = _registry(write_effectful=False, read_effectful=False)
    seal = seal_for(_spec("write", "read"), None, r)
    assert seal.effectful_tools == ()


def test_effectful_tools_all_when_all_effectful():
    r = _registry(write_effectful=True, read_effectful=True)
    seal = seal_for(_spec("write", "read"), None, r)
    assert seal.effectful_tools == ("read", "write")  # sorted


def test_effectful_tools_sorted():
    r = ToolRegistry()
    r.register_tool("z_tool", _fn, effectful=True)
    r.register_tool("a_tool", _fn, effectful=True)
    seal = seal_for(_spec("z_tool", "a_tool"), None, r)
    assert seal.effectful_tools == ("a_tool", "z_tool")


# ---------------------------------------------------------------------------
# drift detection
# ---------------------------------------------------------------------------

def test_effectful_flip_produces_drift():
    """Flipping a tool from effectful=True to effectful=False is detected as drift."""
    r_original = _registry(write_effectful=True)
    r_resume = _registry(write_effectful=False)
    original = seal_for(_spec("write", "read"), None, r_original)
    resumed = seal_for(_spec("write", "read"), None, r_resume)
    assert "effectful_tools" in diff_seals(original, resumed)


def test_no_drift_when_effectful_unchanged():
    r = _registry()
    a = seal_for(_spec("write", "read"), None, r)
    b = seal_for(_spec("write", "read"), None, r)
    assert "effectful_tools" not in diff_seals(a, b)


# ---------------------------------------------------------------------------
# backward-compat: old seal (effectful_tools absent / defaulted to ()) diffs
# against a live seal with non-empty effectful_tools
# ---------------------------------------------------------------------------

def test_old_seal_empty_diffs_against_live_non_empty():
    """A seal from before this field was added has effectful_tools=().
    If the live pipeline has effectful tools, the diff catches it."""
    r = _registry(write_effectful=True)
    live = seal_for(_spec("write", "read"), None, r)
    # Simulate a seal written before this field existed: effectful_tools=()
    old = live.model_copy(update={"effectful_tools": ()})
    # live has "write" in effectful_tools; old has (); drift expected
    assert "effectful_tools" in diff_seals(old, live)


def test_old_seal_empty_matches_live_empty():
    """If the live pipeline has no effectful tools either, no drift."""
    r = _registry(write_effectful=False, read_effectful=False)
    live = seal_for(_spec("write", "read"), None, r)
    old = live.model_copy(update={"effectful_tools": ()})
    assert "effectful_tools" not in diff_seals(old, live)


# ---------------------------------------------------------------------------
# fail-closed: missing tool ref treated as effectful
# ---------------------------------------------------------------------------

def test_missing_tool_treated_as_effectful():
    """A ref declared on the spec but absent from the registry is fail-closed True."""
    r = ToolRegistry()
    r.register_tool("known", _fn, effectful=False)
    # spec declares "unknown" which is not in the registry
    seal = seal_for(_spec("known", "unknown"), None, r)
    assert "unknown" in seal.effectful_tools
    assert "known" not in seal.effectful_tools


def test_all_missing_tools_treated_as_effectful():
    r = ToolRegistry()  # empty
    seal = seal_for(_spec("tool_a", "tool_b"), None, r)
    assert seal.effectful_tools == ("tool_a", "tool_b")
