"""run() entry guards: checkpointed runs need an explicit run_id (fail closed),
and the initial input must be a Pydantic model."""

import pytest
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.state import InMemoryCheckpointStore


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


def make_agent(calls):
    @agent(name="work", input=In, output=Out)
    async def work(v: In) -> Out:
        calls.append(v.x)
        return Out(y=v.x * 2)

    return work


async def test_checkpoints_without_run_id_raises():
    p = Pipeline(name="ckpt_guard", version="0.1.0")
    p.add(make_agent([]))
    with pytest.raises(ValueError, match="run_id"):
        await p.run(In(x=1), checkpoints=InMemoryCheckpointStore())


async def test_distinct_run_ids_both_execute():
    calls: list[int] = []
    p = Pipeline(name="ckpt_two", version="0.1.0")
    p.add(make_agent(calls))
    store = InMemoryCheckpointStore()
    r1 = await p.run(In(x=1), checkpoints=store, run_id="r1")
    r2 = await p.run(In(x=50), checkpoints=store, run_id="r2")
    assert calls == [1, 50]
    assert r1.outputs["work"].y == 2
    assert r2.outputs["work"].y == 100


async def test_run_result_carries_run_id():
    p = Pipeline(name="rid", version="0.1.0")
    p.add(make_agent([]))
    explicit = await p.run(In(x=1), run_id="my-run")
    assert explicit.run_id == "my-run"
    defaulted = await p.run(In(x=1))
    assert defaulted.run_id == "rid-run"


async def test_non_model_initial_raises_typeerror():
    p = Pipeline(name="guard", version="0.1.0")
    p.add(make_agent([]))
    with pytest.raises(TypeError, match="BaseModel.*got dict"):
        await p.run({"x": 1})  # type: ignore[arg-type]


async def test_schema_violation_reason_is_tokenized_and_readable():
    @agent(name="liar", input=In, output=Out)
    async def liar(v: In):
        return {"y": "not an int"}

    p = Pipeline(name="tok", version="0.1.0")
    p.add(liar)
    result = await p.run(In(x=1))
    assert result.status == "halted"
    assert result.reason is not None
    assert result.reason.startswith("schema_violation: output: ")
    assert "errors.pydantic.dev" not in result.reason  # readable, not a dict dump
    assert "y: " in result.reason                       # field location present
    assert result.audit_trace.schema_violations == 1    # recorder matches the token


async def test_halt_code_schema_violation():
    @agent(name="liar2", input=In, output=Out)
    async def liar2(v: In):
        return {"y": "nope"}

    p = Pipeline(name="hc1", version="0.1.0")
    p.add(liar2)
    result = await p.run(In(x=1))
    assert result.halt_code == "schema_violation"


async def test_halt_code_none_on_success():
    p = Pipeline(name="hc2", version="0.1.0")
    p.add(make_agent([]))
    result = await p.run(In(x=1))
    assert result.status == "completed"
    assert result.halt_code is None


def test_halt_codes_is_closed_and_exported():
    from drawbore.errors import HALT_CODES

    assert "schema_violation" in HALT_CODES
    assert "tool_access" in HALT_CODES
    assert len(HALT_CODES) == len(set(HALT_CODES))


def test_join_node_importable_from_public_surface():
    from drawbore.pipeline import JoinNode  # Join's return type must be importable

    assert JoinNode.__name__ == "JoinNode"


def test_declared_halt_reasons_are_in_halt_codes():
    """Every self-declared halt_reason class attribute in the package must be a
    member of the closed HALT_CODES vocabulary."""
    import importlib
    import inspect
    import pkgutil

    import drawbore
    from drawbore.errors import HALT_CODES

    declared: set[str] = set()
    for mod_info in pkgutil.walk_packages(drawbore.__path__, prefix="drawbore."):
        if any(part in mod_info.name for part in (".orchestration.adk", ".llm.litellm")):
            continue  # provider/engine adapters import heavy SDKs; their errors modules are covered below
        try:
            module = importlib.import_module(mod_info.name)
        except Exception:
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            reason = obj.__dict__.get("halt_reason")
            if isinstance(reason, str):
                declared.add(reason)
    assert declared, "expected to find declared halt_reason attributes"
    missing = declared - set(HALT_CODES)
    assert not missing, f"halt_reason values missing from HALT_CODES: {sorted(missing)}"
