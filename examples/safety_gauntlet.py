# examples/safety_gauntlet.py
"""Regression-test that your pipeline's safety layer still refuses the canonical attacks.

Run it in CI (e.g. `python examples/safety_gauntlet.py`). A case returns a `Containment`
verdict; `None` means the attack was NOT contained — a failure.
"""
import asyncio

from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.testing import schema_violation, assert_contained


class Txn(BaseModel):
    id: str


class Decision(BaseModel):
    decision: str


@agent(name="decider", input=Txn, output=Decision, model="m-1.0")
async def decider(v: Txn) -> Decision: ...   # one-shot model agent


async def main() -> None:
    pipeline = Pipeline("demo").add(decider)
    # If the model emits an output that violates the schema, Drawbore must halt — prove it:
    case = schema_violation("decider", {"not": "a decision"})
    await assert_contained(pipeline, case, initial=Txn(id="t-1"))
    print("Containment holds: the safety layer refused the schema-violating output.")


if __name__ == "__main__":
    asyncio.run(main())
