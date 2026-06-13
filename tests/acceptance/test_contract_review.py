"""Acceptance suite: contract clause-review pipeline.

ONE self-contained acceptance suite: small Pydantic models, five narrow agents, a
``build_contract_pipeline(...)`` helper, a pre-populated ToolRegistry, and an injected
in-memory evidence store — run through the REAL safety layer via ``pipeline.test_mode``.
Every non-first step uses explicit ``From(...)`` bindings. The taint breaker is the
centerpiece: an untrusted uploaded contract must never reach an exfil-capable sink,
even when the model-driven loop tries to send it there.
"""

import json

import pytest
from pydantic import BaseModel
from typing import Literal

from drawbore.agent import agent
from drawbore.config import AgentCatalog, ConfigResolutionError, from_json, to_json
from drawbore.escalation import EscalationPolicy, HasConfidence
from drawbore.evidence import (
    EVIDENCE_TOOL_REF,
    EvidenceHandle,
    EvidencePolicy,
    InMemoryEvidenceStore,
    register_evidence_tool,
)
from drawbore.pipeline import Pipeline
from drawbore.pipeline.binding import From
from drawbore.testing import TestingError, call, final
from drawbore.tools import ToolRegistry


# --- locked refs ---
REF_LOADER    = "examples.contract_review.document_loader"
REF_EXTRACTOR = "examples.contract_review.clause_extractor"
REF_RISK      = "examples.contract_review.risk_assessor"
REF_VERIFIER  = "examples.contract_review.party_verifier"
REF_REDLINE   = "examples.contract_review.redline_writer"

CONTRACT_SOURCE = "contracts_upload.fetch_document"   # mcp / untrusted upload source
EXTERNAL_SHARE  = "external_share.post"               # exfil-capable external sink
# EVIDENCE_TOOL_REF is imported ("evidence://retrieve").

SEED_HANDLE = "seed_clause_evidence"   # test FIXTURE only (not a framework promise)


# --- domain models (small; legible to non-engineers) ---
class ContractInput(BaseModel):
    document_id: str


class ContractDocument(BaseModel):
    document_id: str
    party_a: str
    party_b: str
    clause_rows: list[dict]        # structured clauses (large -> compressible)
    review_evidence_handle: str    # seeded fixture field for the retrieval branch


class ClauseReviewInput(BaseModel):
    review_evidence_handle: str
    party_a: str
    party_b: str


class ClauseReviewResult(BaseModel):
    clauses_ok: bool
    clauses_reviewed: int
    notes: str


class RiskInput(BaseModel):
    party_a: str
    party_b: str
    clause_rows: list[dict]


class RiskAssessment(BaseModel, HasConfidence):
    risk_level: str     # "low" / "medium" / "high"
    confidence: float
    factors: list[str]


class PartyVerifyInput(BaseModel):
    party_a: str
    party_b: str


class PartyVerification(BaseModel):
    parties_identified: bool
    party_count: int


class RedlineInput(BaseModel):
    clauses_ok: bool
    risk_level: str
    parties_identified: bool


class RedlineResult(BaseModel):
    decision: Literal["approve", "redline", "escalate"]
    reasons: list[str]


# --- the pre-populated registry (all tool refs present BEFORE from_json) ---
async def _unmocked(args):
    raise AssertionError("real tool handler must not run in test mode")


def build_contract_registry(store: InMemoryEvidenceStore) -> ToolRegistry:
    """Build the ToolRegistry the live pipeline AND ``from_json`` resolve against.

    All declared tool refs are registered up front so resolution never fails late.
    In test mode every handler is replaced by a mock (or, for ``evidence://retrieve``,
    rebound to the test store) — the real handlers never run."""
    reg = ToolRegistry()
    reg.register_mcp_tool(
        CONTRACT_SOURCE, _unmocked,
        allowed_operations=("invoke",), schema={"type": "object"},
    )
    reg.register_tool(
        EXTERNAL_SHARE, _unmocked,
        allowed_operations=("invoke",), schema={"type": "object"},
        exfil_capable=True,
    )
    register_evidence_tool(reg, store=store)
    return reg


def seed_review_store() -> InMemoryEvidenceStore:
    """An in-memory evidence store seeded with the clause-evidence handle the
    retrieval branch reads. allow_search=True / allow_full=False makes the allow/deny
    retrieval cases differ by retrieval MODE on the same seeded policy."""
    store = InMemoryEvidenceStore()
    handle = EvidenceHandle(
        handle_id=SEED_HANDLE, run_id="seed", step=0, source_agent="seed",
        content_type="application/json", original_hash="sha256:seed-orig",
        compressed_hash="sha256:seed-comp", original_tokens=300, compressed_tokens=30,
        transform="json_rows",
    )
    store.put(
        handle,
        original=[{"id": i, "clause": f"clause_{i}"} for i in range(3)],
        compressed=[],
    )
    store.set_policy(SEED_HANDLE, allow_full=False, allow_search=True)
    return store


def test_registry_has_all_declared_tool_refs():
    reg = build_contract_registry(seed_review_store())
    assert reg.has(CONTRACT_SOURCE)
    assert reg.has(EXTERNAL_SHARE)
    assert reg.has(EVIDENCE_TOOL_REF)


def test_models_validate_a_canonical_shape():
    doc = ContractDocument(
        document_id="doc_001",
        party_a="Acme Corp",
        party_b="Widget Inc",
        clause_rows=[{"id": 0, "clause": "clause_0"}],
        review_evidence_handle=SEED_HANDLE,
    )
    assert doc.party_a == "Acme Corp" and doc.review_evidence_handle == SEED_HANDLE


# --- the five agents (each narrow; receives only its schema-declared fields) ---
@agent(name="document_loader", input=ContractInput, output=ContractDocument,
       tools=[CONTRACT_SOURCE])
async def document_loader(v: ContractInput, tools) -> ContractDocument:
    row = await tools.call(CONTRACT_SOURCE, {"document_id": v.document_id})
    return ContractDocument(**row)


@agent(name="clause_extractor", input=ClauseReviewInput, output=ClauseReviewResult,
       model="clause-extractor-1.0", tools=[EVIDENCE_TOOL_REF, CONTRACT_SOURCE, EXTERNAL_SHARE])
async def clause_extractor(v: ClauseReviewInput, tools) -> ClauseReviewResult: ...
# model+tools loop; fn body unused


@agent(name="risk_assessor", input=RiskInput, output=RiskAssessment,
       model="risk-assessor-1.0")
async def risk_assessor(v: RiskInput) -> RiskAssessment: ...
# model one-shot; fn body unused


@agent(name="party_verifier", input=PartyVerifyInput, output=PartyVerification)
async def party_verifier(v: PartyVerifyInput) -> PartyVerification:
    return PartyVerification(parties_identified=True, party_count=2)


@agent(name="redline_writer", input=RedlineInput, output=RedlineResult)
async def redline_writer(v: RedlineInput) -> RedlineResult:
    reasons: list[str] = []
    if not v.parties_identified:
        return RedlineResult(decision="redline", reasons=["parties_not_identified"])
    if v.risk_level == "high":
        reasons.append("high_risk")
    if not v.clauses_ok:
        reasons.append("clause_review_failed")
    if reasons:
        return RedlineResult(decision="redline", reasons=reasons)
    return RedlineResult(decision="approve", reasons=["all_checks_passed"])


# --- risk-assessor evidence policy (opt-in compression; original retained) ---
CLAUSE_EVIDENCE_POLICY = EvidencePolicy(
    name="clause_evidence", enabled=True, mode="compress",
    allowed_transforms=("json_rows",), min_tokens=1,
    allow_search_retrieval=True, allow_full_retrieval=False,
)


def build_contract_pipeline(registry, *, on_failure=None):
    """Build the five-step contract clause-review pipeline + AgentCatalog.

    Topology: document_loader -> {clause_extractor, risk_assessor, party_verifier}
    -> redline_writer. ONLY the first step relies on initial-input behavior; every
    later step declares explicit From(...) bindings. Returns (pipeline, catalog)."""
    pipeline = Pipeline(
        name="contract_review", version="1.0.0", registry=registry,
        confidence_threshold=0.8, on_failure=on_failure,
    )
    pipeline.add(document_loader)   # first step: initial input
    pipeline.add(
        clause_extractor,
        inputs={
            "review_evidence_handle": From("document_loader.review_evidence_handle"),
            "party_a": From("document_loader.party_a"),
            "party_b": From("document_loader.party_b"),
        },
        depends_on=["document_loader"],
    )
    pipeline.add(
        risk_assessor,
        inputs={
            "party_a": From("document_loader.party_a"),
            "party_b": From("document_loader.party_b"),
            "clause_rows": From("document_loader.clause_rows"),
        },
        depends_on=["document_loader"],
        evidence=CLAUSE_EVIDENCE_POLICY,
    )
    pipeline.add(
        party_verifier,
        inputs={
            "party_a": From("document_loader.party_a"),
            "party_b": From("document_loader.party_b"),
        },
        depends_on=["document_loader"],
    )
    pipeline.add(
        redline_writer,
        inputs={
            "clauses_ok": From("clause_extractor.clauses_ok"),
            "risk_level": From("risk_assessor.risk_level"),
            "parties_identified": From("party_verifier.parties_identified"),
        },
        depends_on=["clause_extractor", "risk_assessor", "party_verifier"],
    )
    catalog = AgentCatalog()
    catalog.register(REF_LOADER, document_loader)
    catalog.register(REF_EXTRACTOR, clause_extractor)
    catalog.register(REF_RISK, risk_assessor)
    catalog.register(REF_VERIFIER, party_verifier)
    catalog.register(REF_REDLINE, redline_writer)
    return pipeline, catalog


# --- canonical mocks (controlled externals only) ---
def _canonical_contract_document():
    return {
        "document_id": "doc_001",
        "party_a": "Acme Corp",
        "party_b": "Widget Inc",
        "clause_rows": [
            {"id": i, "clause": f"clause_{i}", "risk_score": i % 3, "note": "x" * 10}
            for i in range(40)
        ],
        "review_evidence_handle": SEED_HANDLE,
    }


def canonical_mocks():
    """Happy-path mock bundle: the contract document for CONTRACT_SOURCE, the one-shot
    risk response by agent name, and the clause_extractor loop script."""
    return dict(
        mock_tools={
            CONTRACT_SOURCE: _canonical_contract_document(),
        },
        mock_model_responses={
            "risk_assessor": {
                "risk_level": "low", "confidence": 0.92, "factors": ["standard_terms"],
            },
        },
        mock_loop_scripts={
            "clause_extractor": [
                # TODO(retrieval-as-evidence): adopt EvidenceBundle when the governed-retrieval design lands
                call(EVIDENCE_TOOL_REF, {"handle_id": SEED_HANDLE, "mode": "search", "query": ""}),
                final({"clauses_ok": True, "clauses_reviewed": 3, "notes": "clauses consistent with submission"}),
            ],
        },
    )


def _step(trace, agent_name):
    matches = [s for s in trace.step_records if s.agent == agent_name]
    if not matches:
        raise AssertionError(
            f"no step record for agent '{agent_name}' in trace "
            f"(available: {[s.agent for s in trace.step_records]})"
        )
    return matches[0]


# ============================================================
# Acceptance cases
# ============================================================

async def test_happy_path_completes():   # acceptance case 1
    store = seed_review_store()
    registry = build_contract_registry(store)
    pipeline, _catalog = build_contract_pipeline(registry)
    async with pipeline.test_mode(evidence_store=store, **canonical_mocks()) as test:
        result = await test.run(ContractInput(document_id="doc_001"))

    assert result.status == "completed"
    redline = result.outputs["redline_writer"]
    assert isinstance(redline, RedlineResult)
    assert redline.decision == "approve"
    trace = result.audit_trace
    assert trace.status == "completed"
    assert trace.steps == 5
    assert result.escalations == [] and trace.escalations == 0
    assert trace.schema_violations == 0


async def test_audit_trace_is_compliance_readable():   # acceptance case 2
    store = seed_review_store()
    registry = build_contract_registry(store)
    pipeline, _ = build_contract_pipeline(registry)
    async with pipeline.test_mode(evidence_store=store, **canonical_mocks()) as test:
        result = await test.run(ContractInput(document_id="doc_001"))

    text = result.audit_trace.legible()
    # pipeline identity and final status
    assert "contract_review" in text
    assert "v1.0.0" in text
    assert "completed" in text
    # summary counts
    assert "Steps completed: 5" in text
    assert "Escalations: 0" in text
    assert "Schema violations: 0" in text
    # tool calls from deterministic and model-loop steps
    assert f"{CONTRACT_SOURCE} (invoke) -> ok" in text    # document_loader
    assert f"{EVIDENCE_TOOL_REF} (invoke) -> ok" in text  # clause_extractor loop
    # model turns recorded for model-backed steps
    assert "turns:" in text
    # evidence compression decision visible on the risk step
    assert "evidence compressed" in text
    assert "handle " in text
    # each agent named in the trace
    for name in ("document_loader", "clause_extractor", "risk_assessor",
                 "party_verifier", "redline_writer"):
        assert name in text, f"{name!r} not found in legible() output"


async def test_evidence_compression_recorded():   # acceptance case 3
    store = seed_review_store()
    registry = build_contract_registry(store)
    pipeline, _ = build_contract_pipeline(registry)
    async with pipeline.test_mode(evidence_store=store, **canonical_mocks()) as test:
        result = await test.run(ContractInput(document_id="doc_001"))

    assert result.status == "completed"
    risk_step = _step(result.audit_trace, "risk_assessor")
    # the decision is on the model step and names a stored handle
    assert risk_step.evidence is not None
    assert "compressed" in risk_step.evidence
    assert "handle " in risk_step.evidence
    # the original is retained in the injected store under the compression handle
    handle_id = risk_step.evidence.split("handle ")[-1].split(";")[0].split()[0].strip()
    assert handle_id != SEED_HANDLE          # distinct from the seeded clause-review handle
    assert store.metadata(handle_id).handle_id == handle_id


async def test_governed_retrieval_allowed_through_proxy():   # acceptance case 4 (allow)
    store = seed_review_store()                  # allow_search=True, allow_full=False
    registry = build_contract_registry(store)
    pipeline, _ = build_contract_pipeline(registry)
    async with pipeline.test_mode(evidence_store=store, **canonical_mocks()) as test:
        result = await test.run(ContractInput(document_id="doc_001"))

    assert result.status == "completed"
    extractor_step = _step(result.audit_trace, "clause_extractor")
    # the in-loop retrieval went through the proxy and is audited as ok
    assert any(f"{EVIDENCE_TOOL_REF} (invoke) -> ok" in tc for tc in extractor_step.tool_calls)


async def test_full_retrieval_denied_fails_closed():   # acceptance case 5 (deny)
    store = seed_review_store()                  # allow_full=False
    registry = build_contract_registry(store)
    pipeline, _ = build_contract_pipeline(registry)
    mocks = canonical_mocks()
    # Deny case: the loop asks for FULL retrieval, which the seeded policy forbids.
    mocks["mock_loop_scripts"]["clause_extractor"] = [
        call(EVIDENCE_TOOL_REF, {"handle_id": SEED_HANDLE, "mode": "full"}),
        final({"clauses_ok": True, "clauses_reviewed": 0, "notes": "should not reach"}),
    ]
    async with pipeline.test_mode(evidence_store=store, **mocks) as test:
        result = await test.run(ContractInput(document_id="doc_001"))

    assert result.status == "halted"
    assert result.halted_at == "clause_extractor"
    extractor_step = _step(result.audit_trace, "clause_extractor")
    assert extractor_step.status != "ok"
    # the forbidden full retrieval was attempted and logged as an error
    assert any(f"{EVIDENCE_TOOL_REF} (invoke) -> error" in tc for tc in extractor_step.tool_calls)
    # downstream never ran
    assert "clause_extractor" not in result.outputs
    assert "redline_writer" not in result.outputs


async def test_taint_breaker_blocks_exfil_of_untrusted_contract():   # acceptance case 6 (CENTERPIECE)
    """The taint breaker: an agent reads from the untrusted contract upload
    (CONTRACT_SOURCE, registered mcp/untrusted), tainting its step scope. Any
    subsequent attempt to call an exfil-capable sink (EXTERNAL_SHARE) in the
    same step is refused by the proxy with a taint_violation. Drawbore halts
    before the sink fires — redline_writer never runs."""
    store = seed_review_store()
    registry = build_contract_registry(store)
    pipeline, _ = build_contract_pipeline(registry)
    mocks = canonical_mocks()
    # Add the exfil sink to mock_tools so the loop script can reach it (it is still
    # refused by the taint breaker; the mock is never called).
    mocks["mock_tools"][EXTERNAL_SHARE] = {"ok": True}
    # Override the clause_extractor loop: first call CONTRACT_SOURCE (taints the step),
    # then attempt to forward to EXTERNAL_SHARE (blocked by the taint breaker).
    mocks["mock_loop_scripts"]["clause_extractor"] = [
        call(CONTRACT_SOURCE, {"document_id": "doc_001"}),
        call(EXTERNAL_SHARE, {"content": "full contract text"}),
        final({"clauses_ok": True, "clauses_reviewed": 0, "notes": "should not reach"}),
    ]
    async with pipeline.test_mode(evidence_store=store, **mocks) as test:
        result = await test.run(ContractInput(document_id="doc_001"))

    assert result.status in ("halted", "escalated"), f"expected halt; got {result.status!r}"
    assert result.reason.startswith("taint_violation:"), (
        f"expected taint_violation; got {result.reason!r}"
    )
    assert result.halted_at == "clause_extractor"
    assert "redline_writer" not in result.outputs


async def test_low_confidence_escalates():   # acceptance case 7
    store = seed_review_store()
    registry = build_contract_registry(store)
    # Variant: configure an escalation policy so a low-confidence trigger dispatches
    # (status "escalated"). The canonical happy-path build leaves on_failure=None,
    # so case 1 stays zero-escalation.
    pipeline, _ = build_contract_pipeline(
        registry,
        on_failure=EscalationPolicy(channel="legal_review", target="counsel", mode="sync"),
    )
    mocks = canonical_mocks()
    mocks["mock_model_responses"]["risk_assessor"] = {
        "risk_level": "high", "confidence": 0.2, "factors": ["non_standard_liability"],
    }
    async with pipeline.test_mode(evidence_store=store, **mocks) as test:
        result = await test.run(ContractInput(document_id="doc_001"))

    assert result.status == "escalated"
    assert result.halted_at == "risk_assessor"
    assert "confidence_below_threshold" in result.reason
    assert len(result.escalations) == 1
    assert result.audit_trace.escalations == 1
    assert "confidence_below_threshold" in result.audit_trace.reason
    # blast radius: document_loader and its peers may run before halt; redline never does
    assert "redline_writer" not in result.outputs


async def test_schema_violation_halts():   # acceptance case 8
    store = seed_review_store()
    registry = build_contract_registry(store)
    pipeline, _ = build_contract_pipeline(registry)
    mocks = canonical_mocks()
    # risk_assessor returns a dict missing the required 'risk_level' -> output schema violation.
    mocks["mock_model_responses"]["risk_assessor"] = {"confidence": 0.92, "factors": []}
    async with pipeline.test_mode(evidence_store=store, **mocks) as test:
        result = await test.run(ContractInput(document_id="doc_001"))

    assert result.status == "halted"
    assert result.halted_at == "risk_assessor"
    assert result.reason.startswith("schema_violation: output:")
    assert result.audit_trace.schema_violations == 1
    # downstream never ran
    assert "redline_writer" not in result.outputs


async def test_unmocked_tool_fails_closed():   # acceptance case 9
    store = seed_review_store()
    registry = build_contract_registry(store)
    pipeline, _ = build_contract_pipeline(registry)
    mocks = canonical_mocks()
    # Drop the contract source mock: document_loader declares it but it is unmocked.
    del mocks["mock_tools"][CONTRACT_SOURCE]
    async with pipeline.test_mode(evidence_store=store, **mocks) as test:
        result = await test.run(ContractInput(document_id="doc_001"))

    assert result.status == "halted"
    assert result.halted_at == "document_loader"
    assert result.reason.startswith("testing_error")
    loader_step = _step(result.audit_trace, "document_loader")
    assert loader_step.status != "ok"
    assert any(CONTRACT_SOURCE in tc for tc in loader_step.tool_calls)
    assert result.outputs == {}


async def test_config_round_trip_and_drift():   # acceptance case 10
    store = seed_review_store()
    registry = build_contract_registry(store)
    pipeline, catalog = build_contract_pipeline(registry)
    manifest = to_json(pipeline, agents=catalog)

    # manifest is shape only — no mock data, no store, no seeded handles
    assert SEED_HANDLE not in manifest
    assert "Acme Corp" not in manifest
    assert "clause-extractor-1.0" in manifest   # model name IS shape (sanity check)

    rebuilt = from_json(manifest, agents=catalog, registry=registry)
    async with rebuilt.test_mode(evidence_store=store, **canonical_mocks()) as test:
        result = await test.run(ContractInput(document_id="doc_001"))

    assert result.status == "completed"
    assert result.steps_run == 5
    redline = result.outputs["redline_writer"]
    assert isinstance(redline, RedlineResult)
    assert redline.decision == "approve"

    # drift: mutate the first agent's version; from_json must reject it
    data = json.loads(manifest)
    data["agents"][0]["version"] = "9.9.9"
    with pytest.raises(ConfigResolutionError, match="version drift"):
        from_json(data, agents=catalog, registry=registry)
