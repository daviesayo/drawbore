# tests/acceptance/test_safety_gauntlet.py
import pytest
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.escalation import EscalationPolicy, HasConfidence
from drawbore.tools import ToolRegistry
from drawbore.testing import (
    Containment, run_pack, run_containment,
    schema_violation, low_confidence, unmocked_tool,
    breaker_trip, parallel_tool_calls, non_json_final,
)

TOOL_A = "gauntlet:a"
TOOL_B = "gauntlet:b"
SYNC = EscalationPolicy(channel="human", target="ops", mode="sync")


class GIn(BaseModel):
    id: str


class Score(BaseModel, HasConfidence):
    risk_level: str
    confidence: float


class Review(BaseModel):
    ok: bool


@agent(name="scorer", input=GIn, output=Score, model="m-1.0")
async def scorer(v: GIn) -> Score: ...   # one-shot HasConfidence model agent


@agent(name="reviewer", input=GIn, output=Review, model="m-1.0", tools=[TOOL_A, TOOL_B])
async def reviewer(v: GIn, tools) -> Review: ...   # model+tools loop agent


def _confidence_pipeline():
    # confidence_threshold + SYNC policy: a hard halt surfaces as status="escalated"
    # (the realistic default), exercising the mapping's class-before-escalated logic.
    return Pipeline("gauntlet", confidence_threshold=0.8, on_failure=SYNC).add(scorer)


def _loop_pipeline():
    reg = ToolRegistry()

    async def _dummy(args):
        return {"ok": True}

    reg.register_tool(TOOL_A, _dummy, allowed_operations=("invoke",), schema={"type": "object"})
    reg.register_tool(TOOL_B, _dummy, allowed_operations=("invoke",), schema={"type": "object"})
    return Pipeline("gauntlet", registry=reg, confidence_threshold=0.8, on_failure=SYNC).add(reviewer)


async def test_full_pack_is_contained_on_a_policy_bearing_pipeline():
    # one-shot cases (scorer)
    conf = _confidence_pipeline()
    conf_cases = [
        schema_violation("scorer", {"risk_level": "low"}),          # missing 'confidence' -> SCHEMA_REJECT
        low_confidence("scorer", {"risk_level": "high", "confidence": 0.2}),  # 0.2 < 0.8 -> ESCALATED
    ]
    conf_report = await run_pack(conf, conf_cases, initial=GIn(id="t1"))
    assert conf_report["schema_violation:scorer"] is Containment.SCHEMA_REJECT
    assert conf_report["low_confidence:scorer"] is Containment.ESCALATED

    # loop cases (reviewer)
    loop = _loop_pipeline()
    loop_cases = [
        unmocked_tool("reviewer", TOOL_A),
        breaker_trip("reviewer", TOOL_A),
        parallel_tool_calls("reviewer", TOOL_A, TOOL_B),
        non_json_final("reviewer"),
    ]
    loop_report = await run_pack(loop, loop_cases, initial=GIn(id="t2"))
    assert loop_report["unmocked_tool:reviewer"] is Containment.FAIL_CLOSED
    assert loop_report["breaker_trip:reviewer"] is Containment.DENIED_BREAKER
    assert loop_report["parallel_tool_calls:reviewer"] is Containment.MODEL_REFUSED
    assert loop_report["non_json_final:reviewer"] is Containment.MODEL_REFUSED


async def test_regression_bites_a_weakened_guarantee_flips_to_none():
    # Simulate "the schema check was weakened": bad_output is actually schema-VALID,
    # so the run COMPLETES and the case is NOT contained. This proves the gauntlet can
    # go red (the gauntlet soundness tripwire).
    conf = _confidence_pipeline()
    weakened = schema_violation("scorer", {"risk_level": "low", "confidence": 0.95})  # valid Score
    assert await run_containment(conf, weakened, initial=GIn(id="t3")) is None
