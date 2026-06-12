"""Quantitative run metrics on RunResult: per-step duration, per-step token usage
and provider-reported cost (None when the provider does not supply it), and the
tool-proxy call log exposed with a serializer."""

import json

from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.llm import TokenUsage
from drawbore.pipeline import Pipeline
from drawbore.pipeline.binding import From
from drawbore.tools import ToolRegistry


class TxIn(BaseModel):
    transaction_id: str


class Fetched(BaseModel):
    transaction_id: str
    amount: int


class ScoreIn(BaseModel):
    amount: int


class Scored(BaseModel):
    amount: int
    risk: str


def _build():
    reg = ToolRegistry()

    async def real_db(args):
        raise AssertionError("real tool must not run in test mode")

    reg.register_tool("db.read", real_db, allowed_operations=("invoke",),
                      schema={"type": "object"})

    @agent(name="fetch", input=TxIn, output=Fetched, tools=["db.read"])
    async def fetch(v: TxIn, tools) -> Fetched:
        row = await tools.call("db.read", {"id": v.transaction_id})
        return Fetched(transaction_id=v.transaction_id, amount=row["amount"])

    @agent(name="score", input=ScoreIn, output=Scored, model="fake")
    async def score(v: ScoreIn) -> Scored: ...

    p = Pipeline(name="aml", registry=reg)
    p.add(fetch)
    p.add(score, inputs={"amount": From("fetch.amount")}, depends_on=["fetch"])
    return p


async def test_run_metrics_carry_per_step_duration_and_tool_log():
    p = _build()
    async with p.test_mode(
        mock_tools={"db.read": {"amount": 100}},
        mock_model_responses={"score": {"amount": 100, "risk": "low"}},
        mock_model_usage={"score": TokenUsage(input_tokens=12, output_tokens=8, total_tokens=20)},
    ) as test:
        result = await test.run(TxIn(transaction_id="t1"))

    assert result.status == "completed"
    metrics = result.metrics
    assert metrics is not None
    assert metrics.run_id == result.run_id

    # Per-step duration: one StepMetric per recorded step, each with a real,
    # non-negative wall-clock duration.
    assert [s.agent for s in metrics.steps] == ["fetch", "score"]
    for s in metrics.steps:
        assert isinstance(s.duration_seconds, float)
        assert s.duration_seconds >= 0.0

    # Token usage surfaces when the (fake) provider reports it.
    score_metric = next(s for s in metrics.steps if s.agent == "score")
    assert score_metric.tokens == TokenUsage(input_tokens=12, output_tokens=8, total_tokens=20)
    # Cost is None when the provider does not supply it (never fabricated).
    assert score_metric.cost is None
    # The deterministic step has no model, so no token usage.
    fetch_metric = next(s for s in metrics.steps if s.agent == "fetch")
    assert fetch_metric.tokens is None

    # The proxy call log is exposed: the db.read tool call appears with its
    # operation, a real duration, and its disposition.
    assert len(metrics.tool_calls) == 1
    call = metrics.tool_calls[0]
    assert call.tool == "db.read"
    assert call.operation == "invoke"
    assert call.result == "ok"
    assert isinstance(call.duration_seconds, float) and call.duration_seconds >= 0.0

    # The whole metrics object serializes to JSON-safe primitives.
    blob = metrics.to_dict()
    json.dumps(blob)  # must not raise
    assert blob["run_id"] == result.run_id
    assert blob["tool_calls"][0]["tool"] == "db.read"
    score_blob = next(s for s in blob["steps"] if s["agent"] == "score")
    assert score_blob["tokens"] == {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
    assert score_blob["cost"] is None


async def test_token_usage_is_none_when_the_provider_does_not_report_it():
    p = _build()
    async with p.test_mode(
        mock_tools={"db.read": {"amount": 5}},
        mock_model_responses={"score": {"amount": 5, "risk": "low"}},
        # no mock_model_usage -> the fake provider reports no usage
    ) as test:
        result = await test.run(TxIn(transaction_id="t2"))

    assert result.status == "completed"
    score_metric = next(s for s in result.metrics.steps if s.agent == "score")
    assert score_metric.tokens is None
    assert score_metric.cost is None
