from pydantic import BaseModel
from drawbore import agent, Pipeline, From


class Seed(BaseModel):
    value: int


class ADoubled(BaseModel):
    doubled: int


class BLabel(BaseModel):
    label: str


class CIn(BaseModel):
    doubled: int
    label: str


class CResult(BaseModel):
    summary: str


@agent(input=Seed, output=ADoubled)
async def a(value: Seed) -> ADoubled:
    return ADoubled(doubled=value.value * 2)


@agent(input=ADoubled, output=BLabel)
async def b(value: ADoubled) -> BLabel:
    return BLabel(label=f"d{value.doubled}")


@agent(input=CIn, output=CResult)
async def c(value: CIn) -> CResult:
    return CResult(summary=f"{value.doubled}/{value.label}")


async def test_fan_in_pipeline_end_to_end():
    p = Pipeline(name="fanin")
    p.add(a)
    p.add(b)
    p.add(c, inputs={"doubled": From("a.doubled"), "label": From("b.label")})

    result = await p.run(Seed(value=5))

    assert result.status == "completed"
    assert result.steps_run == 3
    assert result.outputs["c"] == CResult(summary="10/d10")
