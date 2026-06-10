import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.evidence import EVIDENCE_TOOL_REF, EvidenceRetrievalError, InMemoryEvidenceStore, register_evidence_tool
from drawbore.pipeline import Pipeline
from drawbore.testing import TestingError
from drawbore.testing.tools import build_scoped_registry


class In(BaseModel):
    x: int


class Out(BaseModel):
    x: int


def _agent(name, tools):
    @agent(name=name, input=In, output=Out, tools=tools)
    async def fn(v: In, tools) -> Out:
        return Out(x=v.x)
    return fn


def _pipeline_with(tool_ref, *, allowed=("invoke",), kind="custom", store=None):
    from drawbore.tools import ToolRegistry
    reg = ToolRegistry()
    async def original(args):
        return {"origin": "real"}
    if kind == "builtin":
        reg.register_builtin(tool_ref, original, allowed_operations=allowed, schema={"type": "object"})
    else:
        reg.register_tool(tool_ref, original, allowed_operations=allowed, schema={"type": "object"})
    p = Pipeline(name="p", registry=reg)
    p.add(_agent("a", [tool_ref]))
    return p


async def test_mock_handler_replaces_original_and_preserves_declaration():
    p = _pipeline_with("svc.read", allowed=("invoke", "read"))
    store = InMemoryEvidenceStore()
    overlay = build_scoped_registry(
        p, mock_tools={"svc.read": {"mocked": True}}, allow_real_tools=(), evidence_store=store,
    )
    tool = overlay.get("svc.read")
    assert tool.allowed_operations == ("invoke", "read")   # preserved
    assert tool.schema == {"type": "object"}               # preserved
    assert await tool.handler({"v": 1}) == {"mocked": True}


async def test_static_sync_async_and_sequence_mocks():
    p = _pipeline_with("svc.read")
    store = InMemoryEvidenceStore()

    def sync_mock(args):
        return {"sync": args["v"]}

    async def async_mock(args):
        return {"async": args["v"]}

    # static value
    ov1 = build_scoped_registry(p, mock_tools={"svc.read": 7}, allow_real_tools=(), evidence_store=store)
    assert await ov1.get("svc.read").handler({"v": 1}) == 7
    # sync callable
    ov2 = build_scoped_registry(p, mock_tools={"svc.read": sync_mock}, allow_real_tools=(), evidence_store=store)
    assert await ov2.get("svc.read").handler({"v": 2}) == {"sync": 2}
    # async callable
    ov3 = build_scoped_registry(p, mock_tools={"svc.read": async_mock}, allow_real_tools=(), evidence_store=store)
    assert await ov3.get("svc.read").handler({"v": 3}) == {"async": 3}
    # sequence consumed in order
    ov4 = build_scoped_registry(p, mock_tools={"svc.read": [10, 20]}, allow_real_tools=(), evidence_store=store)
    h = ov4.get("svc.read").handler
    assert await h({}) == 10 and await h({}) == 20
    with pytest.raises(TestingError, match="exhausted"):
        await h({})


async def test_sequence_with_an_exception_raises_it_through_the_handler():
    p = _pipeline_with("svc.read")
    store = InMemoryEvidenceStore()
    ov = build_scoped_registry(
        p, mock_tools={"svc.read": [RuntimeError("boom")]}, allow_real_tools=(), evidence_store=store,
    )
    with pytest.raises(RuntimeError, match="boom"):
        await ov.get("svc.read").handler({})


async def test_unmocked_declared_tool_is_fail_closed_at_invoke():
    p = _pipeline_with("svc.read")
    store = InMemoryEvidenceStore()
    overlay = build_scoped_registry(p, mock_tools={}, allow_real_tools=(), evidence_store=store)
    # The tool stays VISIBLE (proxy/JIT path still exercised) but its handler refuses
    # to call the outside world.
    assert overlay.has("svc.read")
    with pytest.raises(TestingError, match="no mock"):
        await overlay.get("svc.read").handler({})


async def test_allow_real_tools_uses_the_original_handler():
    p = _pipeline_with("svc.read")
    store = InMemoryEvidenceStore()
    overlay = build_scoped_registry(p, mock_tools={}, allow_real_tools=("svc.read",), evidence_store=store)
    assert await overlay.get("svc.read").handler({}) == {"origin": "real"}


async def test_mock_for_an_undeclared_tool_fails_closed():
    p = _pipeline_with("svc.read")
    store = InMemoryEvidenceStore()
    with pytest.raises(TestingError, match="no pipeline step declares"):
        build_scoped_registry(p, mock_tools={"other.tool": 1}, allow_real_tools=(), evidence_store=store)


async def test_evidence_retrieve_is_rebound_to_the_test_store():
    # The pipeline registry binds evidence://retrieve to store A; the overlay must
    # rebind it to the test store (B) with no duplicate-registration error.
    from drawbore.tools import ToolRegistry
    store_a = InMemoryEvidenceStore()
    reg = ToolRegistry()
    register_evidence_tool(reg, store=store_a)
    p = Pipeline(name="p", registry=reg)
    p.add(_agent("a", [EVIDENCE_TOOL_REF]))
    store_b = InMemoryEvidenceStore()
    overlay = build_scoped_registry(p, mock_tools={}, allow_real_tools=(), evidence_store=store_b)
    tool = overlay.get(EVIDENCE_TOOL_REF)
    assert tool.kind == "builtin"
    # store B (empty) fails closed on an unknown handle — proving B, not A, is bound.
    with pytest.raises(EvidenceRetrievalError, match="unknown evidence handle"):
        await tool.handler({"handle_id": "nope", "mode": "search"})
