import pytest
from pydantic import BaseModel
from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.state import InMemoryCheckpointStore
from drawbore.tools import ToolRegistry, ToolContext


class Seed(BaseModel):
    value: int


class Mid(BaseModel):
    m: int


class Done(BaseModel):
    d: int


async def test_agent_exception_halts_not_raises():
    @agent(name="boom", input=Seed, output=Mid)
    async def boom(v: Seed) -> Mid:
        raise RuntimeError("kaboom")

    p = Pipeline(name="t")
    p.add(boom)
    result = await p.run(Seed(value=1))
    assert result.status == "halted"
    assert result.halted_at == "boom"
    assert result.reason.startswith("agent_error")


async def test_tool_circuit_breaker_halts_not_raises():
    reg = ToolRegistry()

    async def read(args):
        return {"n": 1}

    reg.register_tool("db.read", read)

    @agent(name="spammer", input=Seed, output=Mid, tools=["db.read"])
    async def spammer(v: Seed, tools: ToolContext) -> Mid:
        for _ in range(4):  # default breaker max is 3
            await tools.call("db.read", {"id": v.value})
        return Mid(m=0)

    p = Pipeline(name="t", registry=reg)
    p.add(spammer)
    result = await p.run(Seed(value=1))
    assert result.status == "halted"
    assert result.halted_at == "spammer"
    assert "circuit_breaker" in result.reason


async def test_resume_skips_completed_steps():
    calls = {"a": 0, "b": 0}

    @agent(name="a", input=Seed, output=Mid)
    async def a(v: Seed) -> Mid:
        calls["a"] += 1
        return Mid(m=v.value)

    @agent(name="b", input=Mid, output=Done)
    async def b(v: Mid) -> Done:
        calls["b"] += 1
        if calls["b"] == 1:
            raise RuntimeError("transient")  # fail the first attempt
        return Done(d=v.m)

    store = InMemoryCheckpointStore()
    p = Pipeline(name="t")
    p.add(a)
    p.add(b)

    r1 = await p.run(Seed(value=5), run_id="run-1", checkpoints=store)
    assert r1.status == "halted" and r1.halted_at == "b"
    assert calls == {"a": 1, "b": 1}

    r2 = await p.run(Seed(value=5), run_id="run-1", checkpoints=store)  # resume
    assert r2.status == "completed"
    assert calls == {"a": 1, "b": 2}  # 'a' skipped via checkpoint; 'b' retried
    assert r2.outputs["b"] == Done(d=5)


async def test_oversized_external_input_halts_with_sanitization():
    # The external initial input is bounded before any step runs.
    class Deep(BaseModel):
        data: dict

    @agent(name="x", input=Deep, output=Deep)
    async def x(v: Deep) -> Deep:
        return v

    nested: dict = {"leaf": 1}
    for _ in range(40):  # deeper than the default max_depth (32)
        nested = {"n": nested}

    p = Pipeline(name="t")
    p.add(x)
    result = await p.run(Deep(data=nested))
    assert result.status == "halted"
    assert result.halted_at is None
    assert result.reason.startswith("sanitization")


async def test_resume_with_explicit_bindings():
    from drawbore.pipeline import From

    calls = {"a": 0, "c": 0}

    class S(BaseModel):
        value: int

    class A(BaseModel):
        doubled: int

    class C(BaseModel):
        doubled: int

    class R(BaseModel):
        out: int

    @agent(name="a", input=S, output=A)
    async def a(v: S) -> A:
        calls["a"] += 1
        return A(doubled=v.value * 2)

    @agent(name="c", input=C, output=R)
    async def c(v: C) -> R:
        calls["c"] += 1
        if calls["c"] == 1:
            raise RuntimeError("transient")
        return R(out=v.doubled)

    store = InMemoryCheckpointStore()
    p = Pipeline(name="t")
    p.add(a)
    p.add(c, inputs={"doubled": From("a.doubled")})

    r1 = await p.run(S(value=5), run_id="r", checkpoints=store)
    assert r1.status == "halted" and r1.halted_at == "c"

    r2 = await p.run(S(value=5), run_id="r", checkpoints=store)  # resume
    assert r2.status == "completed"
    assert calls == {"a": 1, "c": 2}  # 'a' skipped via checkpoint; its output feeds c's binding
    assert r2.outputs["c"] == R(out=10)
