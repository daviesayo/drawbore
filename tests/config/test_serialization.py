import json

import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.config import AgentCatalog, ConfigResolutionError, PipelineConfig
from drawbore.config.serialization import to_config, to_json
from drawbore.pipeline import Pipeline
from drawbore.pipeline.binding import From


class TxIn(BaseModel):
    amount: int


class TxOut(BaseModel):
    amount: int
    risk: str = "low"


def _agent(name, *, inp=TxIn, out=TxOut):
    @agent(name=name, input=inp, output=out)
    async def fn(v):  # type: ignore[no-untyped-def]
        return out(amount=getattr(v, "amount", 0))
    return fn


def _linear():
    a, b = _agent("fetch"), _agent("screen")
    cat = AgentCatalog()
    cat.register("pkg:fetch", a)
    cat.register("pkg:screen", b)
    p = Pipeline(name="aml", version="0.1.0")
    p.add(a)                                   # initial
    p.add(b, depends_on=["fetch"])             # previous (explicit, matches derived)
    return p, cat


def test_linear_export_shape():
    p, cat = _linear()
    cfg = to_config(p, agents=cat)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.schema_version == 1
    assert cfg.pipeline.name == "aml" and cfg.pipeline.version == "0.1.0"
    assert [a.name for a in cfg.agents] == ["fetch", "screen"]
    assert [a.ref for a in cfg.agents] == ["pkg:fetch", "pkg:screen"]
    assert cfg.steps[0].input.source == "initial" and cfg.steps[0].depends_on == []
    assert cfg.steps[1].input.source == "previous" and cfg.steps[1].input.agent == "fetch"
    assert cfg.steps[1].depends_on == ["fetch"]
    # agent declaration carries schema evidence + full SHA-256 hashes
    assert cfg.agents[0].input_schema_hash.startswith("sha256:")
    assert len(cfg.agents[0].input_schema_hash.split(":", 1)[1]) == 64


def test_fan_in_export_shape():
    # Happy path for the bindings branch of _derive_input: a correctly-wired bound
    # step must export as source=None + a bindings map (alias 'from') + matching
    # depends_on. Isolates the BindingConfig.model_validate({"from": ...}) call.
    class FetchOut(BaseModel):
        amount: int

    class MergeIn(BaseModel):
        amount: int

    a = _agent("fetch", out=FetchOut)

    @agent(name="merge", input=MergeIn, output=MergeIn)
    async def merge(v: MergeIn) -> MergeIn:
        return v

    cat = AgentCatalog()
    cat.register("pkg:fetch", a)
    cat.register("pkg:merge", merge)
    p = Pipeline(name="aml")
    p.add(a)
    p.add(merge, inputs={"amount": From("fetch.amount")}, depends_on=["fetch"])

    cfg = to_config(p, agents=cat)
    s = cfg.steps[1]
    assert s.input.source is None and s.input.agent is None
    assert s.input.bindings is not None
    assert s.input.bindings["amount"].from_ == "fetch.amount"
    assert s.depends_on == ["fetch"]
    # the bindings branch survives canonical JSON round-trip (alias rendered as 'from')
    data = json.loads(to_json(p, agents=cat))
    assert data["steps"][1]["input"]["bindings"]["amount"] == {"from": "fetch.amount"}


def test_export_uncataloged_agent_fails_closed():
    a = _agent("fetch")
    p = Pipeline(name="aml")
    p.add(a)
    with pytest.raises(ConfigResolutionError, match="not in the agent mapping|no unique catalog ref"):
        to_config(p, agents={})           # empty mapping → uncataloged


def test_export_rejects_a_previous_step_whose_depends_on_lies():
    # Live non-first empty-input step whose depends_on is NOT exactly the predecessor.
    a, b = _agent("fetch"), _agent("screen")
    cat = AgentCatalog()
    cat.register("pkg:fetch", a)
    cat.register("pkg:screen", b)
    p = Pipeline(name="aml")
    p.add(a)
    p.add(b, depends_on=[])               # lying: empty-input non-first must depend on 'fetch'
    with pytest.raises(ConfigResolutionError, match="immediate predecessor"):
        to_config(p, agents=cat)


def test_export_rejects_a_bound_step_whose_depends_on_lies():
    class FanOut(BaseModel):
        amount: int

    class FanIn(BaseModel):
        amount: int

    a = _agent("fetch", out=FanOut)

    @agent(name="merge", input=FanIn, output=FanIn)
    async def merge(v: FanIn) -> FanIn:
        return v

    cat = AgentCatalog()
    cat.register("pkg:fetch", a)
    cat.register("pkg:merge", merge)
    p = Pipeline(name="aml")
    p.add(a)
    # bound step, but depends_on does NOT match the binding source 'fetch'
    p.add(merge, inputs={"amount": From("fetch.amount")}, depends_on=[])
    with pytest.raises(ConfigResolutionError, match="binding sources"):
        to_config(p, agents=cat)


def test_to_json_is_deterministic():
    p, cat = _linear()
    assert to_json(p, agents=cat) == to_json(p, agents=cat)
    # round-trips through json.loads back into the same model
    data = json.loads(to_json(p, agents=cat))
    assert PipelineConfig.model_validate(data) == to_config(p, agents=cat)
