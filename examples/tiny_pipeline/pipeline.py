"""A tiny two-step deterministic Drawbore pipeline (docs example).

Two typed steps, no model and no tools: normalize a payment amount, then
categorize it. Run the test with:

    pytest examples/tiny_pipeline/test_tiny_pipeline.py
"""

from pydantic import BaseModel

from drawbore import Pipeline, agent


class Payment(BaseModel):
    cents: int


class Normalized(BaseModel):
    dollars: float


class Categorized(BaseModel):
    dollars: float
    tier: str


@agent(name="normalize", input=Payment, output=Normalized)
async def normalize(p: Payment) -> Normalized:
    return Normalized(dollars=p.cents / 100)


@agent(name="categorize", input=Normalized, output=Categorized)
async def categorize(n: Normalized) -> Categorized:
    tier = "large" if n.dollars >= 100 else "small"
    return Categorized(dollars=n.dollars, tier=tier)


def build_pipeline() -> Pipeline:
    """Build the tiny pipeline. `categorize` consumes `normalize`'s output as its
    immediate predecessor (a single-predecessor linear step needs no `From`)."""
    pipeline = Pipeline(name="tiny_payments", version="1.0.0")
    pipeline.add(normalize)
    pipeline.add(categorize)
    return pipeline
