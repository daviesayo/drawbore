"""Acceptance suite: end-to-end LLM provider resolution through ``pipeline.run``.

Proves the provider resolution invariants compose with a fake-but-real runtime
(no provider calls): profile resolution + provider audit, one-shot fallback recorded,
missing-credential ``model_config_error`` halt, and ``LocalEngine`` engine-agnostic
fail-closed for a model agent. The REAL Drawbore safety layer (resolve, fallback,
halt-and-escalate, audit) decides every outcome; only the provider boundary is faked.
"""

import json

import pytest
from pydantic import BaseModel

import litellm.exceptions as e
from drawbore.agent import agent
from drawbore.llm import (
    LLMGateway, LLMRuntime, LLMRuntimeConfig, ModelProfile, ModelResponse,
    ModelTarget, ProviderConfig,
)
from drawbore.orchestration import ADKEngine
from drawbore.pipeline import Pipeline


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


class _Scripted(LLMGateway):
    def __init__(self, results):
        self._r = list(results)
        self.models = []

    async def complete(self, request):
        self.models.append(request.model_chain[0])
        item = self._r.pop(0)
        if isinstance(item, BaseException):
            raise item
        return ModelResponse(output=item, model_used=request.model_chain[0], raw_text=json.dumps(item))


class AllAvailable:
    def has_credential(self, *, provider, credential_env):
        return True


@agent(name="scorer", input=In, output=Out, model="profile:judgment")
async def scorer(v: In) -> Out: ...


def _cfg():
    return LLMRuntimeConfig(
        profiles={"judgment": ModelProfile(targets=(
            ModelTarget(provider="openrouter", model="anthropic/claude-3-5-sonnet"),
            ModelTarget(provider="openai", model="gpt-4o"),
        ))},
        providers={
            "openrouter": ProviderConfig(credential_env="OPENROUTER_API_KEY"),
            "openai": ProviderConfig(credential_env="OPENAI_API_KEY"),
        },
    )


async def test_profile_run_records_provider_audit():
    p = Pipeline(name="p", version="1.0.0")
    p.add(scorer)
    gw = _Scripted([{"y": 1}])
    engine = ADKEngine(llm_runtime=LLMRuntime(config=_cfg(), gateway=gw, credential_checker=AllAvailable()))
    result = await p.run(In(x=1), engine=engine)
    assert result.status == "completed"
    text = result.audit_trace.legible()
    assert "model profile:judgment" in text
    assert "selected" in text


async def test_one_shot_fallback_records_both_attempts():
    p = Pipeline(name="p", version="1.0.0")
    p.add(scorer)
    gw = _Scripted([e.Timeout(message="t", model="m", llm_provider="openrouter"), {"y": 2}])
    engine = ADKEngine(llm_runtime=LLMRuntime(config=_cfg(), gateway=gw, credential_checker=AllAvailable()))
    result = await p.run(In(x=1), engine=engine)
    assert result.status == "completed"
    attempts = result.audit_trace.step_records[0].model.attempts
    assert [a.outcome for a in attempts] == ["fallback", "success"]


async def test_missing_credential_halts_with_model_config_error():
    p = Pipeline(name="p", version="1.0.0")
    p.add(scorer)

    class NoneAvailable:
        def has_credential(self, *, provider, credential_env):
            return False

    engine = ADKEngine(llm_runtime=LLMRuntime(config=_cfg(), gateway=_Scripted([]), credential_checker=NoneAvailable()))
    result = await p.run(In(x=1), engine=engine)
    assert result.status == "halted"
    assert "model_config_error" in result.reason


async def test_local_engine_still_fails_closed_for_model_agent():
    from drawbore.orchestration import LocalEngine
    p = Pipeline(name="p", version="1.0.0")
    p.add(scorer)
    result = await p.run(In(x=1), engine=LocalEngine())
    assert result.status == "halted"
    # Pin the stable halt-reason key, not just the message wording.
    assert result.reason.startswith("engine_error")
    assert "model" in result.reason  # the reason explains a model-path failure...
    assert "adk" not in result.reason.lower()  # ...without ever naming ADK (Forbidden Pattern)
