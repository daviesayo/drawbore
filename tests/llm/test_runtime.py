import json

import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.llm import (
    LLMConfigError,
    LLMError,
    LLMGateway,
    LLMRuntime,
    LLMRuntimeConfig,
    ModelProfile,
    ModelResponse,
    ModelTarget,
    ModelUnavailableError,
    ProviderConfig,
)
import litellm.exceptions as e


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


def _spec(model, fallback_model=None):
    @agent(name="scorer", input=In, output=Out, model=model, fallback_model=fallback_model)
    async def scorer(v: In) -> Out: ...
    return scorer.spec


class _ScriptedGateway(LLMGateway):
    """Returns/raises per call in order, keyed only by call index (the runtime drives
    single-model requests)."""
    def __init__(self, results):
        self._results = list(results)
        self.models = []

    async def complete(self, request):
        self.models.append(request.model_chain[0])
        item = self._results.pop(0)
        if isinstance(item, BaseException):
            raise item
        return ModelResponse(output=item, model_used=request.model_chain[0], raw_text=json.dumps(item))


class AllAvailable:
    def has_credential(self, *, provider, credential_env):
        return True


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


async def test_one_shot_success_records_single_attempt():
    gw = _ScriptedGateway([{"y": 1}])
    rt = LLMRuntime(config=_cfg(), gateway=gw, credential_checker=AllAvailable())
    spec = _spec("profile:judgment")
    chain = rt.resolve(spec)
    resp = await rt.complete(spec, In(x=1), chain)
    assert resp.output == {"y": 1}
    assert resp.audit.selected_model == "anthropic/claude-3-5-sonnet"
    assert [a.outcome for a in resp.audit.attempts] == ["success"]
    assert gw.models == ["openrouter/anthropic/claude-3-5-sonnet"]


async def test_transport_failure_falls_back_and_records_both(monkeypatch):
    gw = _ScriptedGateway([e.Timeout(message="t", model="m", llm_provider="openrouter"), {"y": 2}])
    rt = LLMRuntime(config=_cfg(), gateway=gw, credential_checker=AllAvailable())
    spec = _spec("profile:judgment")
    resp = await rt.complete(spec, In(x=1), rt.resolve(spec))
    assert resp.output == {"y": 2}
    assert [a.outcome for a in resp.audit.attempts] == ["fallback", "success"]
    assert resp.audit.attempts[0].reason == "timeout"
    assert resp.audit.selected_model == "gpt-4o"


async def test_contract_failure_does_not_fall_back():
    gw = _ScriptedGateway([LLMError("non-JSON")])
    rt = LLMRuntime(config=_cfg(), gateway=gw, credential_checker=AllAvailable())
    spec = _spec("profile:judgment")
    with pytest.raises(LLMError):
        await rt.complete(spec, In(x=1), rt.resolve(spec))
    assert gw.models == ["openrouter/anthropic/claude-3-5-sonnet"]  # only ONE attempt


async def test_auth_failure_is_model_config_error():
    gw = _ScriptedGateway([e.AuthenticationError(message="a", model="m", llm_provider="openrouter")])
    rt = LLMRuntime(config=_cfg(), gateway=gw, credential_checker=AllAvailable())
    spec = _spec("profile:judgment")
    with pytest.raises(LLMConfigError):
        await rt.complete(spec, In(x=1), rt.resolve(spec))


async def test_all_transport_failures_exhaust_to_model_unavailable():
    gw = _ScriptedGateway([
        e.Timeout(message="t", model="m", llm_provider="openrouter"),
        e.RateLimitError(message="r", model="m", llm_provider="openai"),
    ])
    rt = LLMRuntime(config=_cfg(), gateway=gw, credential_checker=AllAvailable())
    spec = _spec("profile:judgment")
    with pytest.raises(ModelUnavailableError):
        await rt.complete(spec, In(x=1), rt.resolve(spec))


async def test_from_gateway_runs_direct_strings_without_profiles():
    gw = _ScriptedGateway([{"y": 9}])
    rt = LLMRuntime.from_gateway(gw)
    spec = _spec("gpt-4o")
    resp = await rt.complete(spec, In(x=1), rt.resolve(spec))
    assert resp.output == {"y": 9}
    assert resp.audit.attempts[0].source == "direct"
