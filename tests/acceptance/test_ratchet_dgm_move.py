# tests/acceptance/test_ratchet_dgm_move.py
"""Acceptance: improvement under a safety ratchet.

Four moves against a remittance-shaped baseline: the judge-disabling move is
rejected by corpus replay; a benign tightening admits (with dedup'd, no-op
growth); an authority widening is returned for human review; a tampered corpus
fails closed before any replay.
"""
import dataclasses
import json

import pytest
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
from drawbore.testing.loop import call, final
from drawbore.tools import ToolRegistry

FETCH_TOOL = "remit:fetch"
NOTE_TOOL = "remit:note"
AUDIT_TOOL = "remit:audit"


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


class ReviewIn(BaseModel):
    status: str


class Review(BaseModel):
    evidence_ok: bool
    status: str


class ConfirmationIn(BaseModel):
    evidence_ok: bool


class Confirmation(BaseModel):
    decision: str


@agent(name="transaction_retriever", input=TxIn, output=TxRow, tools=[FETCH_TOOL])
async def transaction_retriever(v: TxIn, tools) -> TxRow:
    row = await tools.call(FETCH_TOOL, {"id": v.transaction_id})
    return TxRow(**row)


@agent(name="risk_scorer", input=RiskIn, output=RiskScore, model="risk-1.0")
async def risk_scorer(v: RiskIn) -> RiskScore: ...        # one-shot model


@agent(name="evidence_reviewer", input=ReviewIn, output=Review,
       model="review-1.0", tools=[NOTE_TOOL])
async def evidence_reviewer(v: ReviewIn, tools) -> Review: ...   # model+tools loop


@agent(name="confirmation_writer", input=ConfirmationIn, output=Confirmation)
async def confirmation_writer(v: ConfirmationIn) -> Confirmation:
    return Confirmation(decision="confirmed")


# A widened variant of the risk scorer: same contract, plus a tool. Used ONLY by
# the widening move's catalog so the resolver passes and the authority layer decides.
@agent(name="risk_scorer", input=RiskIn, output=RiskScore, model="risk-1.0",
       tools=[AUDIT_TOOL])
async def risk_scorer_widened(v: RiskIn, tools) -> RiskScore: ...


def _registry():
    reg = ToolRegistry()

    async def _impl(args):
        return {"amount": 250000, "status": "pending"}

    for ref in (FETCH_TOOL, NOTE_TOOL, AUDIT_TOOL):
        reg.register_tool(ref, _impl, allowed_operations=("invoke",), schema={"type": "object"})
    return reg


def _build(scorer, registry, *, threshold=0.8):
    p = Pipeline("remittance_confirmation", version="1.0.0", registry=registry,
                 confidence_threshold=threshold)
    p.add(transaction_retriever)
    p.add(scorer,
          inputs={"amount": From("transaction_retriever.amount"),
                  "status": From("transaction_retriever.status")},
          depends_on=["transaction_retriever"])
    p.add(evidence_reviewer,
          inputs={"status": From("transaction_retriever.status")},
          depends_on=["transaction_retriever"])
    # fan-out: scorer and reviewer are parallel branches; the writer consumes the reviewer — a compact analogue of the remittance topology, sufficient for what derive_cases observes
    p.add(confirmation_writer,
          inputs={"evidence_ok": From("evidence_reviewer.evidence_ok")},
          depends_on=["evidence_reviewer"])
    catalog = AgentCatalog()
    catalog.register("remit/retriever", transaction_retriever)
    catalog.register("remit/risk_scorer", scorer)
    catalog.register("remit/reviewer", evidence_reviewer)
    catalog.register("remit/writer", confirmation_writer)
    return p, catalog


SERIALIZABLE_MOCKS = {
    "mock_tools": {
        FETCH_TOOL: {"amount": 250000, "status": "pending"},
        NOTE_TOOL: {"noted": True},
    },
    "mock_model_responses": {
        "risk_scorer": {"risk_level": "low", "confidence": 0.95, "factors": ["clean"]},
    },
}
FULL_MOCKS = {
    **SERIALIZABLE_MOCKS,
    "mock_loop_scripts": {
        "evidence_reviewer": [call(NOTE_TOOL, {"q": ""}), final({"evidence_ok": True, "status": "ok"})],
    },
}
INITIAL = TxIn(transaction_id="tx-1")


async def _seed():
    """Run the full baseline (loop script included), derive from the serializable
    subset, return everything the gate needs."""
    registry = _registry()
    pipeline, catalog = _build(risk_scorer, registry)
    config = to_config(pipeline, agents=catalog)
    async with pipeline.test_mode(**FULL_MOCKS) as tp:
        result = await tp.run(INITIAL)
    assert result.status == "completed"
    corpus = InMemoryRegressionCorpus()
    for case in derive_cases(config, pipeline, result, initial=INITIAL,
                             baseline_mocks=SERIALIZABLE_MOCKS,
                             derived_at="2026-06-13T00:00:00Z", predecessor=None):
        corpus.append(case, sponsor="a.reviewer")
    # mechanical derivation: exactly the scorer's two provocations, in order
    assert [c.containment_case.kind for c in corpus.cases()] == [
        "schema_violation", "low_confidence",
    ]
    return registry, catalog, config, corpus


def _manifest_with(config, **pipeline_overrides) -> str:
    doc = config.model_dump(mode="json")
    doc["pipeline"].update(pipeline_overrides)
    return json.dumps(doc)


async def test_the_dgm_move_is_rejected_by_corpus_replay():
    registry, catalog, config, corpus = await _seed()
    sink = InMemoryRatchetSink()
    root = corpus.root()
    candidate = _manifest_with(config, confidence_threshold=None)  # disable the judge
    verdict = await admit(
        candidate, agents=catalog, corpus=corpus, sponsor="a.reviewer",
        baseline_config=config, initial_input=INITIAL,
        baseline_mocks=SERIALIZABLE_MOCKS, derived_at="2026-06-13T01:00:00Z",
        registry=registry, sink=sink,
    )
    assert verdict.admitted is False
    assert verdict.rejection_layer == "corpus"
    low_conf_case_id = corpus.cases()[1].case_id        # the low-confidence property
    assert verdict.failed_case_id == low_conf_case_id
    assert corpus.root() == root                        # root unchanged
    text = verdict.legible()
    assert "REJECTED" in text and low_conf_case_id in text
    assert sink.verdicts == [verdict]


async def test_the_benign_tightening_admits_with_dedup_no_op_growth():
    registry, catalog, config, corpus = await _seed()
    root = corpus.root()
    candidate = _manifest_with(config, confidence_threshold=0.9)   # strict tightening
    verdict = await admit(
        candidate, agents=catalog, corpus=corpus, sponsor="a.reviewer",
        baseline_config=config, initial_input=INITIAL,
        baseline_mocks=SERIALIZABLE_MOCKS, derived_at="2026-06-13T01:00:00Z",
        registry=registry,
    )
    assert verdict.admitted is True
    assert verdict.pipeline is not None
    assert verdict.corpus_root_after is None            # growth dedup'd to a no-op
    assert corpus.root() == root
    # adoption is reference replacement:
    active = verdict.pipeline
    assert active.name == "remittance_confirmation"


async def test_the_widening_move_is_held_for_human_review():
    registry, catalog, config, corpus = await _seed()
    root = corpus.root()
    # candidate manifest: the scorer now declares an extra tool
    doc = config.model_dump(mode="json")
    for a in doc["agents"]:
        if a["name"] == "risk_scorer":
            a["tools"] = [AUDIT_TOOL]
    # the live catalog must agree (resolver drift-check) — register the widened spec
    widened_registry = _registry()
    _pipeline_w, catalog_w = _build(risk_scorer_widened, widened_registry)
    verdict = await admit(
        json.dumps(doc), agents=catalog_w, corpus=corpus, sponsor="a.reviewer",
        baseline_config=config, initial_input=INITIAL,
        baseline_mocks=SERIALIZABLE_MOCKS, derived_at="2026-06-13T01:00:00Z",
        registry=widened_registry,
    )
    assert verdict.admitted is False
    assert verdict.rejection_layer == "authority"
    assert verdict.authority_diff is not None and not verdict.authority_diff.ok
    assert any(f.ref == AUDIT_TOOL for f in verdict.authority_diff.added)
    assert corpus.root() == root
    assert "may call tool" in verdict.legible()


async def test_a_tampered_corpus_fails_closed_before_any_replay():
    registry, catalog, config, corpus = await _seed()
    corpus._cases[0] = dataclasses.replace(
        corpus._cases[0], derived_at="1999-01-01T00:00:00Z"
    )
    verdict = await admit(
        config.model_dump(mode="json"), agents=catalog, corpus=corpus,
        sponsor="a.reviewer", baseline_config=config, initial_input=INITIAL,
        baseline_mocks=SERIALIZABLE_MOCKS, derived_at="2026-06-13T01:00:00Z",
        registry=registry,
    )
    assert verdict.admitted is False
    assert verdict.rejection_layer == "corpus_integrity"
