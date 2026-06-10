"""Risk-based branch + join example. High risk -> enhanced due diligence;
otherwise -> standard review; an exactly_one Join feeds the writer."""
from pydantic import BaseModel
from drawbore import agent, Pipeline, From
from drawbore.pipeline import When, Join


class Txn(BaseModel):
    amount: int

class Risk(BaseModel):
    level: str

class Review(BaseModel):
    verdict: str

class Decision(BaseModel):
    verdict: str


@agent(name="risk", input=Txn, output=Risk)
async def risk(v: Txn) -> Risk:
    return Risk(level="high" if v.amount > 10000 else "low")

@agent(name="enhanced_dd", input=Risk, output=Review)
async def enhanced_dd(v: Risk) -> Review:
    return Review(verdict="enhanced")

@agent(name="standard_review", input=Risk, output=Review)
async def standard_review(v: Risk) -> Review:
    return Review(verdict="standard")

@agent(name="writer", input=Review, output=Decision)
async def writer(v: Review) -> Decision:
    return Decision(verdict=v.verdict)


def build():
    p = Pipeline(name="risk_branch")
    p.add(risk)
    p.add(enhanced_dd, inputs={"level": From("risk.level")}, when=When("risk.level", equals="high"))
    p.add(standard_review, inputs={"level": From("risk.level")}, when=When("risk.level", in_=("low", "medium")))
    p.add(Join("review", sources=["enhanced_dd", "standard_review"], policy="exactly_one", output=Review))
    p.add(writer, inputs={"verdict": From("review.verdict")})
    return p
