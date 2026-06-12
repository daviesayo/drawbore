# tests/ratchet/test_gate.py
import dataclasses
import json

from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.config import AgentCatalog, to_config
from drawbore.escalation import HasConfidence
from drawbore.pipeline.binding import From
from drawbore.ratchet import (
    InMemoryRatchetSink,
    InMemoryRegressionCorpus,
    admit,
    derive_cases,
)
from drawbore.tools import ToolRegistry

FETCH_TOOL = "data:fetch"
AUDIT_TOOL = "audit:log"


class TxIn(BaseModel):
    transaction_id: str


class TxRow(BaseModel):
    amount: int
    status: str


class RiskIn(BaseModel):
    amount: int
    status: str


class RiskScore(BaseModel, HasConfidence):
    risk_level: str
    confidence: float
    factors: list[str]


@agent(name="retriever", input=TxIn, output=TxRow, tools=[FETCH_TOOL])
async def retriever(v: TxIn, tools) -> TxRow:
    row = await tools.call(FETCH_TOOL, {"id": v.transaction_id})
    return TxRow(**row)


@agent(name="risk_scorer", input=RiskIn, output=RiskScore, model="risk-1.0")
async def risk_scorer(v: RiskIn) -> RiskScore: ...


def _registry(extra_tool=False):
    reg = ToolRegistry()

    async def _impl(args):
        return {"amount": 100, "status": "pending"}

    reg.register_tool(FETCH_TOOL, _impl, allowed_operations=("invoke",), schema={"type": "object"})
    if extra_tool:
        reg.register_tool(AUDIT_TOOL, _impl, allowed_operations=("invoke",), schema={"type": "object"})
    return reg


def _pipeline(registry):
    p = Pipeline("gate-demo", version="1.0.0", registry=registry, confidence_threshold=0.8)
    p.add(retriever)
    p.add(risk_scorer,
          inputs={"amount": From("retriever.amount"), "status": From("retriever.status")},
          depends_on=["retriever"])
    catalog = AgentCatalog()
    catalog.register("demo/retriever", retriever)
    catalog.register("demo/risk_scorer", risk_scorer)
    return p, catalog


MOCKS = {
    "mock_tools": {FETCH_TOOL: {"amount": 100, "status": "pending"}},
    "mock_model_responses": {
        "risk_scorer": {"risk_level": "low", "confidence": 0.95, "factors": ["clean"]},
    },
}
INITIAL = TxIn(transaction_id="t1")


async def _seeded():
    registry = _registry()
    pipeline, catalog = _pipeline(registry)
    config = to_config(pipeline, agents=catalog)
    async with pipeline.test_mode(**MOCKS) as tp:
        result = await tp.run(INITIAL)
    corpus = InMemoryRegressionCorpus()
    for case in derive_cases(config, pipeline, result, initial=INITIAL,
                             baseline_mocks=MOCKS,
                             derived_at="2026-06-13T00:00:00Z", predecessor=None):
        corpus.append(case, sponsor="a.reviewer")
    return registry, catalog, config, corpus


async def test_malformed_manifest_rejects_at_resolver_layer_and_writes_sink():
    registry, catalog, config, corpus = await _seeded()
    sink = InMemoryRatchetSink()
    verdict = await admit(
        '{"not": "a manifest"}',
        agents=catalog, corpus=corpus, sponsor="a.reviewer",
        baseline_config=config, initial_input=INITIAL, baseline_mocks=MOCKS,
        derived_at="2026-06-13T01:00:00Z", registry=registry, sink=sink,
    )
    assert verdict.admitted is False
    assert verdict.rejection_layer == "resolver"
    assert verdict.pipeline is None
    assert sink.verdicts == [verdict]


async def test_identity_manifest_admits_with_no_growth():
    registry, catalog, config, corpus = await _seeded()
    root_before = corpus.root()
    candidate = config.model_dump(mode="json")
    verdict = await admit(
        candidate,
        agents=catalog, corpus=corpus, sponsor="a.reviewer",
        baseline_config=config, initial_input=INITIAL, baseline_mocks=MOCKS,
        derived_at="2026-06-13T01:00:00Z", registry=registry,
    )
    assert verdict.admitted is True
    assert verdict.pipeline is not None
    # identical replay inputs -> every derived case is a duplicate -> no growth
    assert verdict.corpus_root_after is None
    assert corpus.root() == root_before


async def test_replay_input_mismatch_rejects_at_corpus_integrity():
    registry, catalog, config, corpus = await _seeded()
    verdict = await admit(
        config.model_dump(mode="json"),
        agents=catalog, corpus=corpus, sponsor="a.reviewer",
        baseline_config=config,
        initial_input=TxIn(transaction_id="DIFFERENT"),   # not the pinned initial
        baseline_mocks=MOCKS,
        derived_at="2026-06-13T01:00:00Z", registry=registry,
    )
    assert verdict.admitted is False
    assert verdict.rejection_layer == "corpus_integrity"


async def test_tampered_corpus_rejects_before_any_replay():
    registry, catalog, config, corpus = await _seeded()
    corpus._cases[0] = dataclasses.replace(corpus._cases[0], derived_at="1999-01-01T00:00:00Z")
    verdict = await admit(
        config.model_dump(mode="json"),
        agents=catalog, corpus=corpus, sponsor="a.reviewer",
        baseline_config=config, initial_input=INITIAL, baseline_mocks=MOCKS,
        derived_at="2026-06-13T01:00:00Z", registry=registry,
    )
    assert verdict.admitted is False
    assert verdict.rejection_layer == "corpus_integrity"
