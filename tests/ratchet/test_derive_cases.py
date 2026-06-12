# tests/ratchet/test_derive_cases.py
import pytest
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore._canon import canonical_fingerprint
from drawbore.config import AgentCatalog, to_config
from drawbore.escalation import EscalationPolicy, HasConfidence
from drawbore.pipeline.binding import From
from drawbore.ratchet import RatchetError, derive_cases, mocks_fingerprint
from drawbore.testing import Containment
from drawbore.tools import ToolRegistry

FETCH_TOOL = "data:fetch"


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


class Decision(BaseModel):
    decision: str


@agent(name="retriever", input=TxIn, output=TxRow, tools=[FETCH_TOOL])
async def retriever(v: TxIn, tools) -> TxRow:
    row = await tools.call(FETCH_TOOL, {"id": v.transaction_id})
    return TxRow(**row)


@agent(name="risk_scorer", input=RiskIn, output=RiskScore, model="risk-1.0")
async def risk_scorer(v: RiskIn) -> RiskScore: ...  # one-shot model; body unused


@agent(name="writer", input=RiskIn, output=Decision)
async def writer(v: RiskIn) -> Decision:
    return Decision(decision="ok")


class LoopOut(BaseModel):
    ok: bool


@agent(name="looper", input=TxIn, output=LoopOut, model="loop-1.0", tools=[FETCH_TOOL])
async def looper(v: TxIn, tools) -> LoopOut: ...  # model+tools loop; body unused


def _registry():
    reg = ToolRegistry()

    async def _fetch(args):
        return {"amount": 100, "status": "pending"}

    reg.register_tool(FETCH_TOOL, _fetch, allowed_operations=("invoke",), schema={"type": "object"})
    return reg


def _pipeline(*, on_failure=None):
    p = Pipeline("ratchet-demo", version="1.0.0", registry=_registry(),
                 confidence_threshold=0.8, on_failure=on_failure)
    p.add(retriever)
    p.add(risk_scorer,
          inputs={"amount": From("retriever.amount"), "status": From("retriever.status")},
          depends_on=["retriever"])
    p.add(writer,
          inputs={"amount": From("retriever.amount"), "status": From("retriever.status")},
          depends_on=["retriever"])
    catalog = AgentCatalog()
    catalog.register("demo/retriever", retriever)
    catalog.register("demo/risk_scorer", risk_scorer)
    catalog.register("demo/writer", writer)
    return p, catalog


MOCKS = {
    "mock_tools": {FETCH_TOOL: {"amount": 100, "status": "pending"}},
    "mock_model_responses": {
        "risk_scorer": {"risk_level": "low", "confidence": 0.95, "factors": ["clean"]},
    },
}


async def _baseline(pipeline):
    async with pipeline.test_mode(**MOCKS) as tp:
        return await tp.run(TxIn(transaction_id="t1"))


async def test_derives_two_pinned_cases_for_the_one_shot_step():
    pipeline, catalog = _pipeline()
    config = to_config(pipeline, agents=catalog)
    result = await _baseline(pipeline)
    initial = TxIn(transaction_id="t1")

    cases = derive_cases(
        config, pipeline, result,
        initial=initial, baseline_mocks=MOCKS,
        derived_at="2026-06-13T00:00:00Z", predecessor=None,
    )

    assert [c.case_id for c in cases] == ["c-001", "c-002"]
    schema_case, conf_case = cases

    # 1. schema_violation: observed output with the FIRST required field removed
    assert schema_case.containment_case.kind == "schema_violation"
    assert schema_case.containment_case.target == "risk_scorer"
    assert "risk_level" not in schema_case.containment_case.payload
    assert schema_case.containment_case.expect is Containment.SCHEMA_REJECT
    assert schema_case.property.dim == "schema_violations"
    assert schema_case.property.assertion == "neq"
    assert schema_case.property.value == "0"

    # 2. low_confidence: observed output with confidence forced to 0.0;
    #    no on_failure policy on this pipeline -> HALTED, not ESCALATED
    assert conf_case.containment_case.kind == "low_confidence"
    assert conf_case.containment_case.payload["confidence"] == 0.0
    assert conf_case.containment_case.payload["risk_level"] == "low"
    assert conf_case.containment_case.expect is Containment.HALTED
    assert conf_case.property.dim == "run_status"
    assert conf_case.property.value == "completed"

    # chain + pins
    assert cases[0].predecessor_hash == "genesis"
    assert cases[1].predecessor_hash == cases[0].case_hash
    assert cases[0].initial_hash == canonical_fingerprint(initial.model_dump(mode="json"))
    assert cases[0].baseline_mocks_hash == mocks_fingerprint(MOCKS)
    assert cases[0].baseline_fingerprint == canonical_fingerprint(config.model_dump(mode="json"))


async def test_on_failure_policy_flips_expected_verdict_to_escalated():
    policy = EscalationPolicy(channel="human", target="ops", mode="sync")
    pipeline, catalog = _pipeline(on_failure=policy)
    config = to_config(pipeline, agents=catalog)
    result = await _baseline(pipeline)
    cases = derive_cases(
        config, pipeline, result,
        initial=TxIn(transaction_id="t1"), baseline_mocks=MOCKS,
        derived_at="2026-06-13T00:00:00Z", predecessor=None,
    )
    conf_case = [c for c in cases if c.containment_case.kind == "low_confidence"][0]
    assert conf_case.containment_case.expect is Containment.ESCALATED


async def test_requires_completed_baseline():
    pipeline, catalog = _pipeline()
    config = to_config(pipeline, agents=catalog)
    # break the baseline: schema-invalid one-shot response -> halted run
    bad = {**MOCKS, "mock_model_responses": {"risk_scorer": {"wrong": "shape"}}}
    async with pipeline.test_mode(**bad) as tp:
        halted = await tp.run(TxIn(transaction_id="t1"))
    assert halted.status != "completed"
    with pytest.raises(RatchetError, match="completed"):
        derive_cases(
            config, pipeline, halted,
            initial=TxIn(transaction_id="t1"), baseline_mocks=MOCKS,
            derived_at="2026-06-13T00:00:00Z", predecessor=None,
        )


async def test_rejects_loop_scripts_in_the_bundle():
    pipeline, catalog = _pipeline()
    config = to_config(pipeline, agents=catalog)
    result = await _baseline(pipeline)
    with pytest.raises(RatchetError, match="mock_loop_scripts"):
        derive_cases(
            config, pipeline, result,
            initial=TxIn(transaction_id="t1"),
            baseline_mocks={**MOCKS, "mock_loop_scripts": {}},
            derived_at="2026-06-13T00:00:00Z", predecessor=None,
        )


async def test_loop_only_pipeline_fails_closed_with_no_derivable_cases():
    # The fail-closed branch of the prefix rule: a pipeline whose ONLY model step
    # is a model+tools loop step yields zero derivable cases — derive_cases must
    # refuse legibly rather than produce a vacuous ratchet.
    from drawbore.testing.loop import final

    reg = _registry()
    p = Pipeline("all-loop", registry=reg, confidence_threshold=0.8)
    p.add(looper)
    cat = AgentCatalog()
    cat.register("demo/looper", looper)
    config = to_config(p, agents=cat)
    # run it to completion via a loop script (full bundle), then derive with the subset
    async with p.test_mode(
        mock_tools={FETCH_TOOL: {"amount": 1, "status": "ok"}},
        mock_loop_scripts={"looper": [final({"ok": True})]},
    ) as tp:
        result = await tp.run(TxIn(transaction_id="t2"))
    assert result.status == "completed"
    with pytest.raises(RatchetError, match="no derivable"):
        derive_cases(
            config, p, result,
            initial=TxIn(transaction_id="t2"),
            baseline_mocks={"mock_tools": {FETCH_TOOL: {"amount": 1, "status": "ok"}}},
            derived_at="2026-06-13T00:00:00Z", predecessor=None,
        )


async def test_prefix_rule_excludes_one_shot_targets_behind_a_loop_step():
    # looper (model+tools) at index 0, one-shot scorer at index 1: the scorer is
    # excluded by the scheduler-order prefix rule, leaving zero cases -> fail closed.
    from drawbore.testing.loop import final

    reg = _registry()
    p = Pipeline("loop-first", registry=reg, confidence_threshold=0.8)
    p.add(looper)

    class GateIn(BaseModel):
        ok: bool

    class GateScore(BaseModel, HasConfidence):
        risk_level: str
        confidence: float

    @agent(name="late_scorer", input=GateIn, output=GateScore, model="risk-1.0")
    async def late_scorer(v: GateIn) -> GateScore: ...

    p.add(late_scorer, inputs={"ok": From("looper.ok")}, depends_on=["looper"])
    cat = AgentCatalog()
    cat.register("demo/looper", looper)
    cat.register("demo/late_scorer", late_scorer)
    config = to_config(p, agents=cat)
    async with p.test_mode(
        mock_tools={FETCH_TOOL: {"amount": 1, "status": "ok"}},
        mock_loop_scripts={"looper": [final({"ok": True})]},
        mock_model_responses={"late_scorer": {"risk_level": "low", "confidence": 0.9}},
    ) as tp:
        result = await tp.run(TxIn(transaction_id="t3"))
    assert result.status == "completed"
    with pytest.raises(RatchetError, match="no derivable"):
        derive_cases(
            config, p, result,
            initial=TxIn(transaction_id="t3"),
            baseline_mocks={"mock_tools": {FETCH_TOOL: {"amount": 1, "status": "ok"}}},
            derived_at="2026-06-13T00:00:00Z", predecessor=None,
        )
