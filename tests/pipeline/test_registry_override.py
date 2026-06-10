import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.tools import ToolRegistry, RunContext


class In(BaseModel):
    x: int


class Out(BaseModel):
    x: int


def _tool_agent(name, tool_ref):
    @agent(name=name, input=In, output=Out, tools=[tool_ref])
    async def fn(v: In, tools) -> Out:
        await tools.call(tool_ref, {"v": v.x})
        return Out(x=v.x)
    return fn


async def test_registry_override_feeds_the_in_process_proxy():
    # The override registry's handler runs (not the pipeline-registry handler),
    # proving ToolProxy was built from the override registry.
    base = ToolRegistry()
    seen = {"base": 0}

    async def base_handler(args):
        seen["base"] += 1
        return {"ok": True}

    base.register_tool("svc.read", base_handler, allowed_operations=("invoke",),
                       schema={"type": "object"})
    p = Pipeline(name="ov", registry=base)
    p.add(_tool_agent("a", "svc.read"))

    override = ToolRegistry()
    seen["override"] = 0

    async def override_handler(args):
        seen["override"] += 1
        return {"ok": True}

    override.register_tool("svc.read", override_handler, allowed_operations=("invoke",),
                           schema={"type": "object"})

    result = await p.run(In(x=1), registry_override=override)
    assert result.status == "completed"
    assert seen["override"] == 1 and seen["base"] == 0


async def test_registry_override_feeds_the_tool_loop_bundle(fake_adk_model, monkeypatch):
    # A model+tools agent builds a ToolLoopBundle from the run registry; with an
    # override the bundle.registry must be the override (so loop tool schemas come
    # from the scoped registry, not the pipeline's). We assert via the bundle the
    # engine receives.
    from drawbore.orchestration import OrchestratorEngine, ToolLoopBundle

    captured = {}

    class CaptureEngine(OrchestratorEngine):
        async def run_step(self, spec, payload, tools=None, *, tool_loop=None):
            captured["bundle"] = tool_loop
            return {"x": payload.x}

    base = ToolRegistry()
    base.register_tool("svc.read", lambda args: {"ok": True},
                       allowed_operations=("invoke",), schema={"type": "object"})
    override = ToolRegistry()
    override.register_tool("svc.read", lambda args: {"ok": True},
                           allowed_operations=("invoke",), schema={"type": "object"})

    @agent(name="m", input=In, output=Out, model="fake", tools=["svc.read"])
    async def m(v: In, tools) -> Out:
        return Out(x=v.x)

    p = Pipeline(name="ov2", registry=base)
    p.add(m)
    await p.run(In(x=2), engine=CaptureEngine(), registry_override=override)
    assert captured["bundle"] is not None
    assert captured["bundle"].registry is override


async def test_no_override_uses_the_pipeline_registry():
    base = ToolRegistry()
    hits = {"n": 0}

    async def h(args):
        hits["n"] += 1
        return {"ok": True}

    base.register_tool("svc.read", h, allowed_operations=("invoke",), schema={"type": "object"})
    p = Pipeline(name="noov", registry=base)
    p.add(_tool_agent("a", "svc.read"))
    result = await p.run(In(x=1))
    assert result.status == "completed" and hits["n"] == 1
