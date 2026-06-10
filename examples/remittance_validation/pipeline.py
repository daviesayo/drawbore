"""A remittance-confirmation pipeline (Drawbore docs example).

A high-stakes pipeline that holds by construction: fetch a transaction through a
narrow tool, score its risk with a confidence-gated model agent, and write a typed
confirmation. Synthetic data only — run it in test mode (no live provider or
database). See the README to run.

This is a simplified, illustrative remodel of the canonical remittance scenario.
The full locked acceptance scenario lives in
tests/acceptance/test_remittance_validation.py.
"""

from typing import Literal

from pydantic import BaseModel

from drawbore import From, Pipeline, agent
from drawbore.escalation import HasConfidence
from drawbore.tools import ToolRegistry

TRANSACTION_TOOL = "transactions_db.read_transaction"


class RemittanceRequest(BaseModel):
    transaction_id: str


class Transaction(BaseModel):
    transaction_id: str
    amount: int          # minor units
    currency: str
    corridor: str        # e.g. "US->PH"
    status: str


class RiskInput(BaseModel):
    amount: int
    currency: str
    corridor: str


class RiskScore(BaseModel, HasConfidence):
    risk_level: str              # "low" / "medium" / "high"
    confidence: float
    factors: list[str]


class ConfirmationInput(BaseModel):
    transaction_id: str
    risk_level: str


class Confirmation(BaseModel):
    transaction_id: str
    decision: Literal["confirmed", "needs_review"]
    risk_level: str


async def _real_transaction_handler(args):
    # In production this reads your transactions database. The example never calls
    # it: test mode supplies a mock. Fail closed if it is ever reached unmocked.
    raise AssertionError("real transaction tool must not run in test mode")


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_tool(
        TRANSACTION_TOOL, _real_transaction_handler,
        allowed_operations=("invoke",), schema={"type": "object"},
    )
    return reg


@agent(name="retrieve_transaction", input=RemittanceRequest, output=Transaction,
       tools=[TRANSACTION_TOOL])
async def retrieve_transaction(req: RemittanceRequest, tools) -> Transaction:
    row = await tools.call(TRANSACTION_TOOL, {"transaction_id": req.transaction_id})
    return Transaction(**row)


@agent(name="score_risk", input=RiskInput, output=RiskScore, model="risk-scorer-1.0")
async def score_risk(v: RiskInput) -> RiskScore: ...  # model one-shot; body unused


@agent(name="write_confirmation", input=ConfirmationInput, output=Confirmation)
async def write_confirmation(v: ConfirmationInput) -> Confirmation:
    decision = "needs_review" if v.risk_level == "high" else "confirmed"
    return Confirmation(
        transaction_id=v.transaction_id, decision=decision, risk_level=v.risk_level,
    )


def build_pipeline(registry: ToolRegistry, *, on_failure=None) -> Pipeline:
    """Build the three-step remittance pipeline. The risk step is confidence-gated
    (threshold 0.8): a low-confidence score halts (or escalates, if `on_failure` is
    set) instead of guessing. Every non-first step declares explicit `From(...)`."""
    pipeline = Pipeline(
        name="remittance_validation", version="1.0.0",
        registry=registry, confidence_threshold=0.8, on_failure=on_failure,
    )
    pipeline.add(retrieve_transaction)
    pipeline.add(
        score_risk,
        inputs={
            "amount": From("retrieve_transaction.amount"),
            "currency": From("retrieve_transaction.currency"),
            "corridor": From("retrieve_transaction.corridor"),
        },
        depends_on=["retrieve_transaction"],
    )
    pipeline.add(
        write_confirmation,
        inputs={
            "transaction_id": From("retrieve_transaction.transaction_id"),
            "risk_level": From("score_risk.risk_level"),
        },
        depends_on=["retrieve_transaction", "score_risk"],
    )
    return pipeline
