"""Unit tests for the per-step resume seal."""

from pydantic import BaseModel

from drawbore.agent.spec import AgentSpec
from drawbore.evidence import EvidencePolicy
from drawbore.state.step_seal import (
    FIELD_LABELS,
    StepSeal,
    diff_seals,
    seal_for,
)


class _In(BaseModel):
    text: str


class _Out(BaseModel):
    verdict: str


class _OutV2(BaseModel):
    verdict: str
    score: float


async def _fn(payload: BaseModel) -> BaseModel:
    return _Out(verdict="ok")


def _spec(**overrides) -> AgentSpec:
    base = dict(
        name="reviewer",
        input=_In,
        output=_Out,
        fn=_fn,
        tools=("search", "fetch"),
        risk_tier="medium",
        version="1.2.0",
        model="openai/gpt-4o-mini",
        fallback_model=None,
        instructions="Review the document.",
    )
    base.update(overrides)
    return AgentSpec(**base)


def test_seal_is_deterministic():
    assert seal_for(_spec(), None) == seal_for(_spec(), None)


def test_seal_is_frozen_and_closed():
    seal = seal_for(_spec(), None)
    import pydantic
    try:
        seal.model = "other"
        raised = False
    except pydantic.ValidationError:
        raised = True
    assert raised


def test_diff_empty_for_identical():
    assert diff_seals(seal_for(_spec(), None), seal_for(_spec(), None)) == ()


def test_model_drift_detected():
    a = seal_for(_spec(), None)
    b = seal_for(_spec(model="openai/gpt-4o"), None)
    assert diff_seals(a, b) == ("model",)


def test_every_sealed_field_drifts_individually():
    base = seal_for(_spec(), None)
    cases = {
        "version": _spec(version="2.0.0"),
        "risk_tier": _spec(risk_tier="high"),
        "requires_human_approval": _spec(requires_human_approval=True),
        "tools": _spec(tools=("fetch", "search")),  # reorder IS drift (parity)
        "model": _spec(model=None),
        "fallback_model": _spec(fallback_model="openai/gpt-4o"),
        "instructions_fingerprint": _spec(instructions="Different."),
        "output_schema_fingerprint": _spec(output=_OutV2),
    }
    for field, mutated_spec in cases.items():
        drifted = diff_seals(base, seal_for(mutated_spec, None))
        assert drifted == (field,), f"{field}: got {drifted}"


def test_input_schema_drift_detected():
    class _In2(BaseModel):
        text: str
        lang: str

    a = seal_for(_spec(), None)
    b = seal_for(_spec(input=_In2), None)
    assert diff_seals(a, b) == ("input_schema_fingerprint",)


def test_evidence_policy_drift_detected():
    a = seal_for(_spec(), EvidencePolicy(enabled=True))
    b = seal_for(_spec(), EvidencePolicy(enabled=True, min_tokens=100))
    assert diff_seals(a, b) == ("evidence_policy_fingerprint",)


def test_no_policy_is_sentinel_and_differs_from_default_policy():
    none_seal = seal_for(_spec(), None)
    policy_seal = seal_for(_spec(), EvidencePolicy())
    assert none_seal.evidence_policy_fingerprint == "none"
    assert diff_seals(none_seal, policy_seal) == ("evidence_policy_fingerprint",)


def test_none_instructions_distinct_from_empty():
    a = seal_for(_spec(instructions=None), None)
    b = seal_for(_spec(instructions=""), None)
    assert a.instructions_fingerprint == "none"
    assert diff_seals(a, b) == ("instructions_fingerprint",)


def test_context_access_stored_verbatim():
    assert seal_for(_spec(), None).context_access == "none"


def test_multi_field_drift_sorted():
    a = seal_for(_spec(), None)
    b = seal_for(_spec(model="x", version="9.9.9"), None)
    assert diff_seals(a, b) == ("model", "version")


def test_friendly_labels_cover_fingerprint_fields():
    assert FIELD_LABELS["instructions_fingerprint"] == "instructions"
    assert FIELD_LABELS["input_schema_fingerprint"] == "input schema"
    assert FIELD_LABELS["output_schema_fingerprint"] == "output schema"
    assert FIELD_LABELS["evidence_policy_fingerprint"] == "evidence policy"
