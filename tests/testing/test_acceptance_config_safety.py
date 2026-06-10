import json

import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.config import AgentCatalog, from_json, to_json, ConfigResolutionError
from drawbore.pipeline import Pipeline
from drawbore.pipeline.binding import From
from drawbore.testing import TestingError
from drawbore.tools import ToolRegistry, registry as global_registry


class TxIn(BaseModel):
    transaction_id: str


class Fetched(BaseModel):
    transaction_id: str
    amount: int


# ScoreIn only carries the field we bind from fetch; the static check requires
# every required input field to be bound when using explicit bindings.
class ScoreIn(BaseModel):
    amount: int


class Scored(BaseModel):
    amount: int
    risk: str


def _config_pipeline():
    reg = ToolRegistry()
    async def real(args):
        raise AssertionError("real handler must not run")
    reg.register_tool("db.read", real, allowed_operations=("invoke",), schema={"type": "object"})

    @agent(name="fetch", input=TxIn, output=Fetched, tools=["db.read"])
    async def fetch(v: TxIn, tools) -> Fetched:
        row = await tools.call("db.read", {"id": v.transaction_id})
        return Fetched(transaction_id=v.transaction_id, amount=row["amount"])

    @agent(name="score", input=ScoreIn, output=Scored, model="fake")
    async def score(v: ScoreIn) -> Scored: ...

    cat = AgentCatalog()
    cat.register("pkg:fetch", fetch)
    cat.register("pkg:score", score)
    p = Pipeline(name="aml", registry=reg)
    p.add(fetch)
    p.add(score, inputs={"amount": From("fetch.amount")}, depends_on=["fetch"])
    return p, cat, reg


async def test_config_loaded_pipeline_runs_in_test_mode():   # spec test 20
    p, cat, reg = _config_pipeline()
    manifest = to_json(p, agents=cat)
    rebuilt = from_json(manifest, agents=cat, registry=reg)   # drift checks happen HERE
    async with rebuilt.test_mode(
        mock_tools={"db.read": {"amount": 5}},
        mock_model_responses={"score": {"amount": 5, "risk": "low"}},
    ) as test:
        result = await test.run(TxIn(transaction_id="t1"))
    assert result.status == "completed" and result.steps_run == 2


async def test_config_drift_is_rejected_before_test_mode():   # spec test 20 (negative)
    p, cat, reg = _config_pipeline()
    data = json.loads(to_json(p, agents=cat))
    data["agents"][0]["version"] = "9.9.9"     # version drift
    with pytest.raises(ConfigResolutionError, match="version drift"):
        from_json(data, agents=cat, registry=reg)   # never reaches test_mode


async def test_no_global_registry_leakage():   # spec test 21
    p, cat, reg = _config_pipeline()
    assert not global_registry.has("db.read")   # precondition
    async with p.test_mode(
        mock_tools={"db.read": {"amount": 1}},
        mock_model_responses={"score": {"amount": 1, "risk": "low"}},
    ) as test:
        await test.run(TxIn(transaction_id="t1"))
    # mocks lived in the scoped overlay only — never installed globally
    assert not global_registry.has("db.read")
    # a second, independent pipeline gets ITS OWN db.read handler (the first run's
    # mock did not leak into it): entering its test mode and running uses the second
    # pipeline's own registry, and the global registry is still clean afterwards.
    p2, _, _ = _config_pipeline()
    async with p2.test_mode(
        mock_tools={"db.read": {"amount": 2}},
        mock_model_responses={"score": {"amount": 2, "risk": "low"}},
    ) as test2:
        r2 = await test2.run(TxIn(transaction_id="t2"))
    assert r2.status == "completed" and r2.outputs["fetch"].amount == 2
    assert p._registry is not p2._registry        # distinct scoped registries, no shared backing
    assert not global_registry.has("db.read")


async def test_no_live_external_calls_by_default():   # spec test 22
    p, cat, reg = _config_pipeline()
    async with p.test_mode(
        # db.read intentionally NOT mocked and NOT in allow_real_tools
        mock_model_responses={"score": {"amount": 1, "risk": "low"}},
    ) as test:
        result = await test.run(TxIn(transaction_id="t1"))
    assert result.status == "halted"
    assert result.halted_at == "fetch"              # halt must occur at fetch, not score
    assert result.reason.startswith("testing_error")   # fail closed, no real call


async def test_allow_real_tools_opts_a_declared_tool_into_its_real_handler():   # spec test 23
    reg = ToolRegistry()
    hits = {"n": 0}
    async def real(args):
        hits["n"] += 1
        return {"amount": 7}
    reg.register_tool("db.read", real, allowed_operations=("invoke",), schema={"type": "object"})

    @agent(name="fetch", input=TxIn, output=Fetched, tools=["db.read"])
    async def fetch(v: TxIn, tools) -> Fetched:
        row = await tools.call("db.read", {"id": v.transaction_id})
        return Fetched(transaction_id=v.transaction_id, amount=row["amount"])

    p = Pipeline(name="aml", registry=reg)
    p.add(fetch)
    async with p.test_mode(allow_real_tools=("db.read",)) as test:
        result = await test.run(TxIn(transaction_id="t1"))
    assert result.status == "completed"
    assert result.outputs["fetch"].amount == 7 and hits["n"] == 1
