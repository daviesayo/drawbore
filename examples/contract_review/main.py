"""Contract clause-review pipeline (Drawbore docs example).

A high-stakes pipeline that holds by construction: load an uploaded contract through
a narrow, untrusted-source tool, review its clauses with a confidence-gated model
agent, and write a typed redline decision. The taint breaker ensures contract content
can never be forwarded to an exfil-capable sink even if the model-driven loop attempts
it. Synthetic data only — run in test mode (no live provider or database).

This is a simplified, illustrative version of the domain scenario. The full locked
acceptance scenario lives in tests/acceptance/test_contract_review.py.
"""
import asyncio
from typing import Literal

from pydantic import BaseModel

from drawbore import From, Pipeline, agent
from drawbore.escalation import HasConfidence
from drawbore.evidence import (
    EVIDENCE_TOOL_REF,
    EvidenceHandle,
    EvidencePolicy,
    InMemoryEvidenceStore,
    register_evidence_tool,
)
from drawbore.testing import call, final
from drawbore.tools import ToolRegistry

CONTRACT_SOURCE = "contracts_upload.fetch_document"   # mcp / untrusted
EXTERNAL_SHARE  = "external_share.post"               # exfil-capable sink
SEED_HANDLE     = "example_clause_evidence"


# --- domain models ---
class ContractRequest(BaseModel):
    document_id: str


class Contract(BaseModel):
    document_id: str
    party_a: str
    party_b: str
    clause_rows: list[dict]
    review_evidence_handle: str


class ClauseInput(BaseModel):
    review_evidence_handle: str
    party_a: str
    party_b: str


class ClauseResult(BaseModel):
    clauses_ok: bool
    clauses_reviewed: int
    notes: str


class RiskInput(BaseModel):
    party_a: str
    party_b: str
    clause_rows: list[dict]


class RiskScore(BaseModel, HasConfidence):
    risk_level: str
    confidence: float
    factors: list[str]


class PartyInput(BaseModel):
    party_a: str
    party_b: str


class PartyResult(BaseModel):
    parties_identified: bool
    party_count: int


class DecisionInput(BaseModel):
    clauses_ok: bool
    risk_level: str
    parties_identified: bool


class Decision(BaseModel):
    decision: Literal["approve", "redline"]
    risk_level: str


async def _unmocked(args):
    raise AssertionError("live handler must not run in test mode")


def build_registry(store: InMemoryEvidenceStore) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_mcp_tool(CONTRACT_SOURCE, _unmocked, schema={"type": "object"})
    reg.register_tool(EXTERNAL_SHARE, _unmocked, schema={"type": "object"}, exfil_capable=True)
    register_evidence_tool(reg, store=store)
    return reg


@agent(name="load_contract", input=ContractRequest, output=Contract, tools=[CONTRACT_SOURCE])
async def load_contract(req: ContractRequest, tools) -> Contract:
    row = await tools.call(CONTRACT_SOURCE, {"document_id": req.document_id})
    return Contract(**row)


@agent(name="review_clauses", input=ClauseInput, output=ClauseResult,
       model="clause-reviewer-1.0", tools=[EVIDENCE_TOOL_REF, CONTRACT_SOURCE, EXTERNAL_SHARE])
async def review_clauses(v: ClauseInput, tools) -> ClauseResult: ...


@agent(name="score_risk", input=RiskInput, output=RiskScore, model="risk-scorer-1.0")
async def score_risk(v: RiskInput) -> RiskScore: ...


@agent(name="verify_parties", input=PartyInput, output=PartyResult)
async def verify_parties(v: PartyInput) -> PartyResult:
    return PartyResult(parties_identified=True, party_count=2)


@agent(name="write_decision", input=DecisionInput, output=Decision)
async def write_decision(v: DecisionInput) -> Decision:
    if v.risk_level == "high" or not v.clauses_ok:
        return Decision(decision="redline", risk_level=v.risk_level)
    return Decision(decision="approve", risk_level=v.risk_level)


RISK_EVIDENCE = EvidencePolicy(
    name="risk_clause_evidence", enabled=True, mode="compress",
    allowed_transforms=("json_rows",), min_tokens=1,
    allow_search_retrieval=True, allow_full_retrieval=False,
)


def build_pipeline(registry: ToolRegistry, *, on_failure=None) -> Pipeline:
    """Five-step contract review pipeline. The risk step is confidence-gated
    (threshold 0.8): a low-confidence assessment halts instead of guessing.
    Every non-first step declares explicit From(...)."""
    pipeline = Pipeline(
        name="contract_review_example", version="1.0.0",
        registry=registry, confidence_threshold=0.8, on_failure=on_failure,
    )
    pipeline.add(load_contract)
    pipeline.add(
        review_clauses,
        inputs={
            "review_evidence_handle": From("load_contract.review_evidence_handle"),
            "party_a": From("load_contract.party_a"),
            "party_b": From("load_contract.party_b"),
        },
        depends_on=["load_contract"],
    )
    pipeline.add(
        score_risk,
        inputs={
            "party_a": From("load_contract.party_a"),
            "party_b": From("load_contract.party_b"),
            "clause_rows": From("load_contract.clause_rows"),
        },
        depends_on=["load_contract"],
        evidence=RISK_EVIDENCE,
    )
    pipeline.add(
        verify_parties,
        inputs={
            "party_a": From("load_contract.party_a"),
            "party_b": From("load_contract.party_b"),
        },
        depends_on=["load_contract"],
    )
    pipeline.add(
        write_decision,
        inputs={
            "clauses_ok": From("review_clauses.clauses_ok"),
            "risk_level": From("score_risk.risk_level"),
            "parties_identified": From("verify_parties.parties_identified"),
        },
        depends_on=["review_clauses", "score_risk", "verify_parties"],
    )
    return pipeline


async def main() -> None:
    store = InMemoryEvidenceStore()
    seed = EvidenceHandle(
        handle_id=SEED_HANDLE, run_id="example", step=0, source_agent="example",
        content_type="application/json", original_hash="sha256:ex-orig",
        compressed_hash="sha256:ex-comp", original_tokens=100, compressed_tokens=10,
        transform="json_rows",
    )
    store.put(seed, original=[{"id": i, "clause": f"ok_{i}"} for i in range(3)], compressed=[])
    store.set_policy(SEED_HANDLE, allow_full=False, allow_search=True)

    registry = build_registry(store)
    pipeline = build_pipeline(registry)

    async with pipeline.test_mode(
        evidence_store=store,
        mock_tools={
            CONTRACT_SOURCE: {
                "document_id": "doc_001",
                "party_a": "Acme Corp",
                "party_b": "Widget Inc",
                "clause_rows": [
                    {"id": i, "clause": f"clause_{i}", "risk_score": i % 3}
                    for i in range(40)
                ],
                "review_evidence_handle": SEED_HANDLE,
            },
        },
        mock_model_responses={
            "score_risk": {"risk_level": "low", "confidence": 0.91, "factors": ["standard_terms"]},
        },
        mock_loop_scripts={
            "review_clauses": [
                call(EVIDENCE_TOOL_REF, {"handle_id": SEED_HANDLE, "mode": "search", "query": ""}),
                final({"clauses_ok": True, "clauses_reviewed": 3, "notes": "clauses consistent"}),
            ],
        },
    ) as test:
        result = await test.run(ContractRequest(document_id="doc_001"))

    print(result.audit_trace.legible())
    assert result.status == "completed"
    print(f"\ndecision: {result.outputs['write_decision'].decision}")


if __name__ == "__main__":
    asyncio.run(main())
