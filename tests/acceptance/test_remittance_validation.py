"""Acceptance suite: the canonical remittance-confirmation validation pipeline.

ONE self-contained acceptance suite: small Pydantic models, five narrow agents, a
`build_remittance_pipeline(...)` helper, a pre-populated ToolRegistry, and an injected
in-memory evidence store — run through the REAL safety layer via `pipeline.test_mode`.
Every non-first step uses explicit `From(...)` bindings. Mocks and the seeded retrieval
handle are supplied OUTSIDE the JSON manifest.
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


# --- locked refs (plan "Locked scenario choices") ---
REF_RETRIEVER = "examples.remittance.transaction_retriever"
REF_IDENTITY = "examples.remittance.identity_validator"
REF_RISK = "examples.remittance.risk_scorer"
REF_REVIEW = "examples.remittance.evidence_reviewer"
REF_CONFIRM = "examples.remittance.confirmation_writer"

TRANSACTION_TOOL = "transactions_db.read_transaction"
IDENTITY_TOOL = "identity_kyc.verify_parties"
# EVIDENCE_TOOL_REF is imported ("evidence://retrieve").

SEED_HANDLE = "seed_review_evidence"  # test/example FIXTURE only (not a framework promise)


# --- data models (small; legible to non-engineers) ---
class TransactionInput(BaseModel):
    transaction_id: str


class Party(BaseModel):
    name: str
    country: str
    account_id: str


class TransactionRecord(BaseModel):
    transaction_id: str
    sender: Party
    recipient: Party
    amount: int                 # minor units
    currency: str
    corridor: str               # e.g. "US->PH"
    status: str                 # e.g. "pending"
    evidence_rows: list[dict]   # structured evidence for the risk model (large -> compressible)
    review_evidence_handle: str # SEEDED fixture field for the retrieval branch


class IdentityCheckInput(BaseModel):
    sender: Party
    recipient: Party


class IdentityResult(BaseModel):
    sender_verified: bool
    recipient_verified: bool
    reason_codes: list[str]


class RiskInput(BaseModel):
    amount: int
    currency: str
    corridor: str
    status: str
    evidence_rows: list[dict]


class RiskScore(BaseModel, HasConfidence):
    risk_level: str             # "low" / "medium" / "high"
    confidence: float
    factors: list[str]


class ReviewInput(BaseModel):
    review_evidence_handle: str
    status: str


class ReviewResult(BaseModel):
    evidence_ok: bool
    rows_reviewed: int
    notes: str


class ConfirmationInput(BaseModel):
    sender_verified: bool
    recipient_verified: bool
    risk_level: str
    evidence_ok: bool


class ConfirmationResult(BaseModel):
    decision: Literal["confirmed", "blocked", "needs_review"]
    reasons: list[str]


# --- the pre-populated import registry (all three tool refs present BEFORE from_json) ---
async def _unmocked(args):
    raise AssertionError("real tool handler must not run in test mode")


def build_import_registry(store: InMemoryEvidenceStore) -> ToolRegistry:
    """Build the ToolRegistry the live pipeline AND `from_json` resolve against.
    All three declared tool refs are registered up front so resolution never fails
    late. In test mode every handler is replaced by a mock (or, for
    `evidence://retrieve`, rebound to the test store) — the real handlers never run."""
    reg = ToolRegistry()
    reg.register_tool(TRANSACTION_TOOL, _unmocked, allowed_operations=("invoke",), schema={"type": "object"})
    reg.register_mcp_tool(IDENTITY_TOOL, _unmocked, allowed_operations=("invoke",), schema={"type": "object"})
    register_evidence_tool(reg, store=store)  # kind="builtin"; rebound to the test store in test mode
    return reg


def seed_review_store() -> InMemoryEvidenceStore:
    """An in-memory evidence store seeded with the review handle the retrieval branch
    reads. allow_search=True / allow_full=False makes the allow/deny retrieval cases
    differ by retrieval MODE on the same seeded policy."""
    store = InMemoryEvidenceStore()
    handle = EvidenceHandle(
        handle_id=SEED_HANDLE, run_id="seed", step=0, source_agent="seed",
        content_type="application/json", original_hash="sha256:seed-orig",
        compressed_hash="sha256:seed-comp", original_tokens=500, compressed_tokens=50,
        transform="json_rows",
    )
    store.put(handle, original=[{"id": i, "rule": f"rule_{i}"} for i in range(3)], compressed=[])
    store.set_policy(SEED_HANDLE, allow_full=False, allow_search=True)
    return store


def test_import_registry_has_all_declared_tool_refs():
    reg = build_import_registry(seed_review_store())
    assert reg.has(TRANSACTION_TOOL)
    assert reg.has(IDENTITY_TOOL)
    assert reg.has(EVIDENCE_TOOL_REF)


def test_models_validate_a_canonical_shape():
    rec = TransactionRecord(
        transaction_id="t", sender=Party(name="A", country="US", account_id="a"),
        recipient=Party(name="B", country="PH", account_id="b"), amount=100,
        currency="USD", corridor="US->PH", status="pending",
        evidence_rows=[{"id": 1}], review_evidence_handle=SEED_HANDLE,
    )
    assert rec.amount == 100 and rec.review_evidence_handle == SEED_HANDLE


# --- the five agents (each narrow; receives only its schema-declared fields) ---
@agent(name="transaction_retriever", input=TransactionInput, output=TransactionRecord,
       tools=[TRANSACTION_TOOL])
async def transaction_retriever(v: TransactionInput, tools) -> TransactionRecord:
    row = await tools.call(TRANSACTION_TOOL, {"transaction_id": v.transaction_id})
    return TransactionRecord(**row)


@agent(name="identity_validator", input=IdentityCheckInput, output=IdentityResult,
       tools=[IDENTITY_TOOL])
async def identity_validator(v: IdentityCheckInput, tools) -> IdentityResult:
    res = await tools.call(
        IDENTITY_TOOL, {"sender": v.sender.model_dump(), "recipient": v.recipient.model_dump()}
    )
    return IdentityResult(**res)


@agent(name="risk_scorer", input=RiskInput, output=RiskScore, model="risk-scorer-1.0")
async def risk_scorer(v: RiskInput) -> RiskScore: ...  # model one-shot; fn body unused


@agent(name="evidence_reviewer", input=ReviewInput, output=ReviewResult,
       model="evidence-reviewer-1.0", tools=[EVIDENCE_TOOL_REF])
async def evidence_reviewer(v: ReviewInput, tools) -> ReviewResult: ...  # model+tools loop; fn body unused


@agent(name="confirmation_writer", input=ConfirmationInput, output=ConfirmationResult)
async def confirmation_writer(v: ConfirmationInput) -> ConfirmationResult:
    reasons: list[str] = []
    if not (v.sender_verified and v.recipient_verified):
        return ConfirmationResult(decision="blocked", reasons=["identity_check_failed"])
    if v.risk_level == "high":
        reasons.append("high_risk")
    if not v.evidence_ok:
        reasons.append("evidence_review_failed")
    if reasons:
        return ConfirmationResult(decision="needs_review", reasons=reasons)
    return ConfirmationResult(decision="confirmed", reasons=["all_checks_passed"])


# --- the risk-scorer evidence policy (opt-in compression; original retained) ---
RISK_EVIDENCE_POLICY = EvidencePolicy(
    name="risk_evidence", enabled=True, mode="compress",
    allowed_transforms=("json_rows",), min_tokens=1,
    allow_search_retrieval=True, allow_full_retrieval=False,
)


def build_remittance_pipeline(registry, *, on_failure=None):
    """Build the canonical five-step remittance pipeline and a populated AgentCatalog.

    Mandatory topology: transaction_retriever -> {identity_validator,
    risk_scorer, evidence_reviewer} -> confirmation_writer. ONLY the first step relies
    on initial-input behavior; every later step declares explicit From(...) bindings.
    Returns (pipeline, catalog)."""
    pipeline = Pipeline(
        name="remittance_confirmation", version="1.0.0", registry=registry,
        confidence_threshold=0.8, on_failure=on_failure,
    )
    pipeline.add(transaction_retriever)  # first step: initial input
    pipeline.add(
        identity_validator,
        inputs={
            "sender": From("transaction_retriever.sender"),
            "recipient": From("transaction_retriever.recipient"),
        },
        depends_on=["transaction_retriever"],
    )
    pipeline.add(
        risk_scorer,
        inputs={
            "amount": From("transaction_retriever.amount"),
            "currency": From("transaction_retriever.currency"),
            "corridor": From("transaction_retriever.corridor"),
            "status": From("transaction_retriever.status"),
            "evidence_rows": From("transaction_retriever.evidence_rows"),
        },
        depends_on=["transaction_retriever"],
        evidence=RISK_EVIDENCE_POLICY,
    )
    pipeline.add(
        evidence_reviewer,
        inputs={
            "review_evidence_handle": From("transaction_retriever.review_evidence_handle"),
            "status": From("transaction_retriever.status"),
        },
        depends_on=["transaction_retriever"],
    )
    pipeline.add(
        confirmation_writer,
        inputs={
            "sender_verified": From("identity_validator.sender_verified"),
            "recipient_verified": From("identity_validator.recipient_verified"),
            "risk_level": From("risk_scorer.risk_level"),
            "evidence_ok": From("evidence_reviewer.evidence_ok"),
        },
        depends_on=["identity_validator", "risk_scorer", "evidence_reviewer"],
    )
    catalog = AgentCatalog()
    catalog.register(REF_RETRIEVER, transaction_retriever)
    catalog.register(REF_IDENTITY, identity_validator)
    catalog.register(REF_RISK, risk_scorer)
    catalog.register(REF_REVIEW, evidence_reviewer)
    catalog.register(REF_CONFIRM, confirmation_writer)
    return pipeline, catalog


# --- canonical mocks (controlled externals only) ---
def _canonical_transaction_row():
    return {
        "transaction_id": "test_123",
        "sender": {"name": "Alice", "country": "US", "account_id": "acct_a"},
        "recipient": {"name": "Bob", "country": "PH", "account_id": "acct_b"},
        "amount": 250000,
        "currency": "USD",
        "corridor": "US->PH",
        "status": "pending",
        "evidence_rows": [
            {"id": i, "rule": f"rule_{i}", "score": i % 7, "note": "x" * 12} for i in range(40)
        ],
        "review_evidence_handle": SEED_HANDLE,
    }


def canonical_mocks():
    """The happy-path mock bundle: tools by ref, the one-shot risk response by agent
    name, and the evidence_reviewer loop script (call retrieve -> final JSON)."""
    return dict(
        mock_tools={
            TRANSACTION_TOOL: _canonical_transaction_row(),
            IDENTITY_TOOL: {"sender_verified": True, "recipient_verified": True, "reason_codes": []},
        },
        mock_model_responses={
            "risk_scorer": {"risk_level": "low", "confidence": 0.95, "factors": ["clean_corridor"]},
        },
        mock_loop_scripts={
            "evidence_reviewer": [
                # empty query matches all seeded rows; intentional for the happy path
                call(EVIDENCE_TOOL_REF, {"handle_id": SEED_HANDLE, "mode": "search", "query": ""}),
                final({"evidence_ok": True, "rows_reviewed": 3, "notes": "evidence consistent"}),
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


async def test_happy_path_completes():   # acceptance case 1
    store = seed_review_store()
    registry = build_import_registry(store)
    pipeline, _catalog = build_remittance_pipeline(registry)
    async with pipeline.test_mode(evidence_store=store, **canonical_mocks()) as test:
        result = await test.run(TransactionInput(transaction_id="test_123"))

    assert result.status == "completed"
    confirmation = result.outputs["confirmation_writer"]
    assert isinstance(confirmation, ConfirmationResult)
    assert confirmation.decision == "confirmed"
    trace = result.audit_trace
    assert trace.status == "completed"
    assert trace.steps == 5
    assert result.escalations == [] and trace.escalations == 0   # happy path: zero escalation
    assert trace.schema_violations == 0


async def test_audit_trace_is_compliance_readable():   # acceptance case 2
    store = seed_review_store()
    registry = build_import_registry(store)
    pipeline, _ = build_remittance_pipeline(registry)
    async with pipeline.test_mode(evidence_store=store, **canonical_mocks()) as test:
        result = await test.run(TransactionInput(transaction_id="test_123"))

    text = result.audit_trace.legible()
    # pipeline identity + final status
    assert "remittance_confirmation" in text
    assert "v1.0.0" in text
    assert "completed" in text
    # summary counts (durable phrases, not a snapshot)
    assert "Steps completed: 5" in text
    assert "Escalations: 0" in text
    assert "Schema violations: 0" in text
    # tool calls (deterministic step + in-loop retrieval) audited
    assert f"{TRANSACTION_TOOL} (invoke) -> ok" in text
    assert f"{IDENTITY_TOOL} (invoke) -> ok" in text
    assert f"{EVIDENCE_TOOL_REF} (invoke) -> ok" in text
    # model turns recorded for the model-backed steps
    assert "turns:" in text
    # evidence compression decision is visible on the risk step
    assert "evidence compressed" in text
    assert "handle " in text
    # each step named
    for name in ("transaction_retriever", "identity_validator", "risk_scorer",
                 "evidence_reviewer", "confirmation_writer"):
        assert name in text, f"{name!r} not found in legible() output"


async def test_evidence_compression_decision_is_recorded():   # acceptance case 3
    store = seed_review_store()
    registry = build_import_registry(store)
    pipeline, _ = build_remittance_pipeline(registry)
    async with pipeline.test_mode(evidence_store=store, **canonical_mocks()) as test:
        result = await test.run(TransactionInput(transaction_id="test_123"))

    assert result.status == "completed"
    risk_step = _step(result.audit_trace, "risk_scorer")
    # the decision is on the model step and names a stored handle
    assert risk_step.evidence is not None
    assert "compressed" in risk_step.evidence
    assert "handle " in risk_step.evidence
    # the ORIGINAL is retained in the injected store under the compression handle.
    # EvidenceDecision.legible() ends with "... handle <handle_id>" (or "... [warnings: ...]").
    handle_id = risk_step.evidence.split("handle ")[-1].split(";")[0].split()[0].strip()
    assert handle_id != SEED_HANDLE                    # distinct from the seeded review handle
    assert store.metadata(handle_id).handle_id == handle_id


async def test_evidence_retrieval_allowed_through_the_proxy():   # acceptance case 4 (allow)
    store = seed_review_store()                    # allow_search=True, allow_full=False
    registry = build_import_registry(store)
    pipeline, _ = build_remittance_pipeline(registry)
    async with pipeline.test_mode(evidence_store=store, **canonical_mocks()) as test:
        result = await test.run(TransactionInput(transaction_id="test_123"))

    assert result.status == "completed"
    review_step = _step(result.audit_trace, "evidence_reviewer")
    # the in-loop retrieval went through the proxy and is audited as ok
    assert any(f"{EVIDENCE_TOOL_REF} (invoke) -> ok" in tc for tc in review_step.tool_calls)
    # Authority-bound retrieval: the canonical loop script issues the retrieve with
    # SEED_HANDLE — the value bound into `review_evidence_handle` from the input field,
    # not a hidden store/audit handle. (The audit tool_calls strings record only
    # `<tool> (<op>) -> <result>`, not args, so the handle value is fixed by the loop
    # script literal above, not asserted here.)


async def test_evidence_full_retrieval_without_policy_fails_closed():   # acceptance case 4 (deny)
    store = seed_review_store()                    # allow_full=False
    registry = build_import_registry(store)
    pipeline, _ = build_remittance_pipeline(registry)
    mocks = canonical_mocks()
    # Deny case: the loop asks for FULL retrieval, which the seeded policy forbids.
    mocks["mock_loop_scripts"]["evidence_reviewer"] = [
        call(EVIDENCE_TOOL_REF, {"handle_id": SEED_HANDLE, "mode": "full"}),
        final({"evidence_ok": True, "rows_reviewed": 0, "notes": "should not reach"}),
    ]
    async with pipeline.test_mode(evidence_store=store, **mocks) as test:
        result = await test.run(TransactionInput(transaction_id="test_123"))

    assert result.status == "halted"
    assert result.halted_at == "evidence_reviewer"
    review_step = _step(result.audit_trace, "evidence_reviewer")
    assert review_step.status != "ok"
    # the forbidden full retrieval was attempted through the proxy and logged as an
    # error (EvidenceRetrievalError is not a ToolError, so the proxy result is "error",
    # not a "denied:*" label) — non-tautological vs the halted_at assertion above
    assert any(f"{EVIDENCE_TOOL_REF} (invoke) -> error" in tc for tc in review_step.tool_calls)
    # the run never produced the reviewer/writer outputs
    assert "evidence_reviewer" not in result.outputs
    assert "confirmation_writer" not in result.outputs


async def test_low_confidence_escalates():   # acceptance case 5 (negative)
    store = seed_review_store()
    registry = build_import_registry(store)
    # Variant: configure an escalation policy so a confidence-trigger DISPATCHES
    # (status "escalated") rather than a plain halt. The canonical happy-path build
    # leaves on_failure=None, so case 1 stays zero-escalation.
    pipeline, _ = build_remittance_pipeline(
        registry,
        on_failure=EscalationPolicy(channel="human_review", target="compliance_ops", mode="sync"),
    )
    mocks = canonical_mocks()
    mocks["mock_model_responses"]["risk_scorer"] = {
        "risk_level": "high", "confidence": 0.2, "factors": ["sanctions_corridor"]
    }
    async with pipeline.test_mode(evidence_store=store, **mocks) as test:
        result = await test.run(TransactionInput(transaction_id="test_123"))

    assert result.status == "escalated"
    assert result.halted_at == "risk_scorer"
    assert "confidence_below_threshold" in result.reason
    assert len(result.escalations) == 1
    # the escalation reason is on the audit record too (legible to a compliance officer)
    assert result.audit_trace.escalations == 1
    assert "confidence_below_threshold" in result.audit_trace.reason
    # blast radius: an earlier parallel branch ran before the halt; nothing downstream did
    assert "identity_validator" in result.outputs
    assert "confirmation_writer" not in result.outputs


async def test_schema_violation_halts_at_the_step():   # acceptance case 6
    store = seed_review_store()
    registry = build_import_registry(store)
    pipeline, _ = build_remittance_pipeline(registry)   # canonical: on_failure=None -> plain halt
    mocks = canonical_mocks()
    # risk_scorer returns a dict missing the required 'risk_level' -> output schema violation.
    mocks["mock_model_responses"]["risk_scorer"] = {"confidence": 0.95, "factors": []}
    async with pipeline.test_mode(evidence_store=store, **mocks) as test:
        result = await test.run(TransactionInput(transaction_id="test_123"))

    assert result.status == "halted"
    assert result.halted_at == "risk_scorer"
    assert result.reason.startswith("schema_violation: output:")
    assert result.audit_trace.schema_violations == 1
    # downstream steps never ran: no reviewer/writer outputs
    assert "evidence_reviewer" not in result.outputs
    assert "confirmation_writer" not in result.outputs
    # step_records are in execution order: only the retriever and its parallel peer
    # identity_validator completed before the halt; the failed risk_scorer is not
    # recorded (no tool calls -> no failed-step record; never succeeded -> no step).
    assert [s.agent for s in result.audit_trace.step_records] == [
        "transaction_retriever", "identity_validator",
    ]


async def test_unmocked_declared_tool_fails_closed():   # acceptance case 7
    store = seed_review_store()
    registry = build_import_registry(store)
    pipeline, _ = build_remittance_pipeline(registry)
    mocks = canonical_mocks()
    # Drop the transaction tool mock: it is declared but unmocked and not allow_real.
    del mocks["mock_tools"][TRANSACTION_TOOL]
    async with pipeline.test_mode(evidence_store=store, **mocks) as test:
        result = await test.run(TransactionInput(transaction_id="test_123"))

    assert result.status == "halted"
    assert result.halted_at == "transaction_retriever"
    assert result.reason.startswith("testing_error")   # fail closed, no live call
    first_step = _step(result.audit_trace, "transaction_retriever")
    assert first_step.status != "ok"
    assert any(TRANSACTION_TOOL in tc for tc in first_step.tool_calls)
    # nothing ran past the first step
    assert result.outputs == {}


async def test_config_round_trip_runs_in_test_mode():   # acceptance case 8
    store = seed_review_store()
    registry = build_import_registry(store)             # registry pre-populated with all 3 tools
    pipeline, catalog = build_remittance_pipeline(registry)
    manifest = to_json(pipeline, agents=catalog)        # catalog supplies symbolic refs

    # the manifest is pipeline SHAPE only — no mocks, no store, no seeded handle, no row data
    assert SEED_HANDLE not in manifest                  # seeded handle is runtime, not shape
    assert "acct_a" not in manifest                     # canonical mock row DATA is not serialized
    assert "risk-scorer-1.0" in manifest                # model name IS shape (sanity check)

    rebuilt = from_json(manifest, agents=catalog, registry=registry)   # drift checks happen HERE
    async with rebuilt.test_mode(evidence_store=store, **canonical_mocks()) as test:
        result = await test.run(TransactionInput(transaction_id="test_123"))
    # the config-loaded pipeline is behaviorally equivalent to the original: it runs all
    # five steps AND produces the same terminal decision (not merely "did not crash")
    assert result.status == "completed"
    assert result.steps_run == 5
    confirmation = result.outputs["confirmation_writer"]
    assert isinstance(confirmation, ConfirmationResult)
    assert confirmation.decision == "confirmed"


async def test_config_drift_is_rejected_before_test_mode():   # acceptance case 8 (negative)
    store = seed_review_store()
    registry = build_import_registry(store)
    pipeline, catalog = build_remittance_pipeline(registry)
    data = json.loads(to_json(pipeline, agents=catalog))
    data["agents"][0]["version"] = "9.9.9"              # version drift on the first agent
    with pytest.raises(ConfigResolutionError, match="version drift"):
        from_json(data, agents=catalog, registry=registry)   # never reaches test_mode
