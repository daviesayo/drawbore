# tests/testing/test_gauntlet_runner_smoke.py
from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.llm import LLMRuntimeConfig, ModelProfile, ModelTarget, ProviderConfig
from drawbore.testing import StaticCredentialChecker
from drawbore.testing.gauntlet import Containment, schema_violation, run_containment, assert_contained, run_pack


class GIn(BaseModel):
    id: str


class Out(BaseModel):
    label: str


@agent(name="scorer", input=GIn, output=Out, model="m-1.0")
async def scorer(v: GIn) -> Out: ...   # one-shot model agent; body unused


@agent(name="pscorer", input=GIn, output=Out, model="profile:judgment")
async def pscorer(v: GIn) -> Out: ...   # profile-bound one-shot model agent; body unused


def _profile_config() -> LLMRuntimeConfig:
    return LLMRuntimeConfig(
        profiles={"judgment": ModelProfile(targets=(
            ModelTarget(provider="openrouter", model="anthropic/claude-3-5-sonnet"),
        ))},
        providers={"openrouter": ProviderConfig(credential_env="OPENROUTER_API_KEY")},
    )


async def test_schema_violation_is_contained_end_to_end():
    pipeline = Pipeline("smoke").add(scorer)
    case = schema_violation("scorer", {"wrong": "field"})   # missing required 'label'
    observed = await run_containment(pipeline, case, initial=GIn(id="x"))
    assert observed is Containment.SCHEMA_REJECT
    await assert_contained(pipeline, case, initial=GIn(id="x"))   # no raise


async def test_not_contained_returns_none_and_assert_raises():
    pipeline = Pipeline("smoke").add(scorer)
    # a schema-VALID output completes -> not contained
    valid = schema_violation("scorer", {"label": "ok"})
    assert await run_containment(pipeline, valid, initial=GIn(id="x")) is None
    import pytest
    with pytest.raises(AssertionError):
        await assert_contained(pipeline, valid, initial=GIn(id="x"))


async def test_profile_bound_pipeline_is_contained_with_llm_config():
    # A profile-bound agent must resolve its model through the runtime config the
    # gauntlet forwards; without it the run degrades to HALTED model_config_error
    # before the attack is provoked.
    pipeline = Pipeline("smoke").add(pscorer)
    case = schema_violation("pscorer", {"wrong": "field"})   # missing required 'label'
    observed = await run_containment(
        pipeline, case, initial=GIn(id="x"),
        llm_config=_profile_config(),
        credential_checker=StaticCredentialChecker(available=True),
    )
    assert observed is Containment.SCHEMA_REJECT
    await assert_contained(
        pipeline, case, initial=GIn(id="x"),
        llm_config=_profile_config(),
        credential_checker=StaticCredentialChecker(available=True),
    )


async def test_run_pack_forwards_llm_config_to_profile_pipeline():
    pipeline = Pipeline("smoke").add(pscorer)
    case = schema_violation("pscorer", {"wrong": "field"})
    report = await run_pack(
        pipeline, [case], initial=GIn(id="x"),
        llm_config=_profile_config(),
        credential_checker=StaticCredentialChecker(available=True),
    )
    assert report[case.name] is Containment.SCHEMA_REJECT
