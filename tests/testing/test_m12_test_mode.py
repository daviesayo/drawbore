import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.llm import LLMRuntimeConfig, ModelProfile, ModelTarget, ProviderConfig
from drawbore.pipeline import Pipeline
from drawbore.testing import StaticCredentialChecker, call, final


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


@agent(name="scorer", input=In, output=Out, model="profile:judgment")
async def scorer(v: In) -> Out: ...


def _config():
    return LLMRuntimeConfig(
        profiles={"judgment": ModelProfile(targets=(
            ModelTarget(provider="openrouter", model="anthropic/claude-3-5-sonnet"),
        ))},
        providers={"openrouter": ProviderConfig(credential_env="OPENROUTER_API_KEY")},
    )


async def test_profile_one_shot_uses_fake_response_no_provider():
    p = Pipeline(name="p", version="1.0.0")
    p.add(scorer)
    async with p.test_mode(
        llm_config=_config(),
        mock_model_responses={"scorer": {"y": 5}},
    ) as test:
        result = await test.run(In(x=1))
    assert result.status == "completed"
    assert result.outputs["scorer"].y == 5
    # the profile resolved AND the model audit recorded a real attempt
    step = result.audit_trace.step_records[0]
    assert step.model.selected_model == "anthropic/claude-3-5-sonnet"


async def test_missing_mock_for_profile_agent_fails_closed():
    p = Pipeline(name="p", version="1.0.0")
    p.add(scorer)
    async with p.test_mode(llm_config=_config(), mock_model_responses={}) as test:
        result = await test.run(In(x=1))
    assert result.status == "halted"
    assert result.reason.startswith("testing_error")


async def test_explicit_missing_credential_raises_model_config_error():
    p = Pipeline(name="p", version="1.0.0")
    p.add(scorer)
    async with p.test_mode(
        llm_config=_config(),
        credential_checker=StaticCredentialChecker(available=False),
        mock_model_responses={"scorer": {"y": 5}},
    ) as test:
        result = await test.run(In(x=1))
    assert result.status == "halted"
    assert "model_config_error" in result.reason
