from pydantic import BaseModel
from drawbore.agent import agent
from drawbore.pipeline import Pipeline, From


class Seed(BaseModel):
    value: int


class ADoubled(BaseModel):
    doubled: int


class BLabel(BaseModel):
    label: str


@agent(input=Seed, output=ADoubled)
async def a(value: Seed) -> ADoubled:
    return ADoubled(doubled=value.value * 2)


@agent(input=ADoubled, output=BLabel)
async def b(value: ADoubled) -> BLabel:
    return BLabel(label=f"d{value.doubled}")


async def test_linear_pipeline_completes():
    p = Pipeline(name="t")
    p.add(a)
    p.add(b)
    result = await p.run(Seed(value=5))
    assert result.status == "completed"
    assert result.steps_run == 2
    assert result.outputs["b"] == BLabel(label="d10")


async def test_output_schema_violation_halts():
    @agent(name="bad", input=Seed, output=ADoubled)
    async def bad(value: Seed) -> ADoubled:
        # returns a dict with a wrong type for `doubled` -> strict validation fails
        return {"doubled": "not-an-int"}

    p = Pipeline(name="t")
    p.add(bad)
    result = await p.run(Seed(value=1))
    assert result.status == "halted"
    assert result.halted_at == "bad"
    assert result.reason.startswith("schema_violation: output:")
    assert result.steps_run == 0


async def test_input_schema_violation_halts():
    # Linear-default binding: predecessor output is structurally missing a
    # required field of the next agent's input -> runtime input violation.
    class AOut(BaseModel):
        x: int

    class BIn(BaseModel):
        y: int

    class BOut(BaseModel):
        ok: bool

    @agent(name="ax", input=Seed, output=AOut)
    async def ax(value: Seed) -> AOut:
        return AOut(x=value.value)

    @agent(name="bx", input=BIn, output=BOut)
    async def bx(value: BIn) -> BOut:
        return BOut(ok=True)

    p = Pipeline(name="t")
    p.add(ax)
    p.add(bx)  # linear default; static check skipped for no-inputs steps
    result = await p.run(Seed(value=1))
    assert result.status == "halted"
    assert result.halted_at == "bx"
    assert result.reason.startswith("schema_violation: input:")
    assert result.steps_run == 1  # ax succeeded, bx failed at its input boundary


async def test_whole_output_binding_runs_end_to_end():
    class XOut(BaseModel):
        a: int
        b: str

    class Detail(BaseModel):
        a: int

    class YIn(BaseModel):
        detail: Detail

    class YOut(BaseModel):
        ok: bool

    @agent(name="x", input=Seed, output=XOut)
    async def x(value: Seed) -> XOut:
        return XOut(a=value.value, b="z")

    @agent(name="y", input=YIn, output=YOut)
    async def y(value: YIn) -> YOut:
        return YOut(ok=value.detail.a == 3)

    p = Pipeline(name="t")
    p.add(x)
    p.add(y, inputs={"detail": From("x")})  # whole-output binding, structural
    result = await p.run(Seed(value=3))
    assert result.status == "completed"
    assert result.outputs["y"] == YOut(ok=True)
