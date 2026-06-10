import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.config import AgentCatalog, ConfigResolutionError, from_json, to_json
from drawbore.pipeline import Pipeline
from drawbore.tools import ToolRegistry


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


async def _noop(args):
    return {}


def _registry():
    reg = ToolRegistry()
    reg.register_tool("t.do", _noop, allowed_operations=("invoke",), schema={"type": "object"})
    return reg


@agent(name="loopy", input=In, output=Out, model="profile:judgment",
       fallback_model="profile:backup", tools=["t.do"])
async def loopy(v: In, tools) -> Out: ...


def test_model_tools_fallback_manifest_now_loads():
    reg = _registry()
    p = Pipeline(name="p", version="1.0.0", registry=reg)
    p.add(loopy)
    catalog = AgentCatalog()
    catalog.register("ex.loopy", loopy)
    manifest = to_json(p, agents=catalog)
    # model + tools + fallback_model previously raised ConfigResolutionError; now it loads
    rebuilt = from_json(manifest, agents=catalog, registry=reg)
    assert rebuilt is not None
    # the declaration strings are preserved exactly (runtime-free)
    assert "profile:judgment" in manifest
    assert "profile:backup" in manifest


def test_drift_is_still_rejected():
    import json
    reg = _registry()
    p = Pipeline(name="p", version="1.0.0", registry=reg)
    p.add(loopy)
    catalog = AgentCatalog()
    catalog.register("ex.loopy", loopy)
    data = json.loads(to_json(p, agents=catalog))
    data["agents"][0]["version"] = "9.9.9"
    with pytest.raises(ConfigResolutionError, match="version drift"):
        from_json(data, agents=catalog, registry=reg)
