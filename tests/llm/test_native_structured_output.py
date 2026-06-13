"""Provider-native structured output: the opt-in reliability fast-path.

These exercise the four touch points — the ``ProviderConfig`` opt-in + collision
guard, ``build_model_request(native=...)`` carrying the output schema, the
resolve-time fail-closed support guard (one-shot only), and the gateway passing
``response_format`` into ``base_kwargs`` so the corrective reask inherits it. The
authoritative ``schema_violation`` gate and ``coerce_structured_output`` are
unchanged — native mode only reduces reprompts, it can never weaken the gate.
"""

import json

import pytest
from pydantic import BaseModel, ValidationError

from drawbore.agent import agent
from drawbore.llm import (
    LLMConfigError,
    LLMRuntime,
    LLMRuntimeConfig,
    ModelProfile,
    ModelRequest,
    ModelTarget,
    ProviderConfig,
)
from drawbore.llm.build import build_model_request
from drawbore.llm.production import ProductionLLMGateway


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


def _spec(model, *, tools=None, fallback_model=None):
    @agent(
        name="scorer", input=In, output=Out, model=model,
        fallback_model=fallback_model, tools=tools,
    )
    async def scorer(v: In) -> Out: ...
    return scorer.spec


class AllAvailable:
    def has_credential(self, *, provider, credential_env):
        return True


def _native_cfg(*, native=True):
    return LLMRuntimeConfig(
        profiles={"judgment": ModelProfile(targets=(
            ModelTarget(provider="anthropic", model="claude-3-5-sonnet-20241022"),
        ))},
        providers={
            "anthropic": ProviderConfig(
                credential_env="ANTHROPIC_API_KEY", native_structured_output=native
            ),
        },
    )


# --- ProviderConfig opt-in + collision guard ---------------------------------

def test_provider_config_native_defaults_false():
    assert ProviderConfig().native_structured_output is False


def test_provider_config_native_round_trips():
    pc = ProviderConfig(native_structured_output=True)
    assert pc.native_structured_output is True


def test_native_with_response_format_in_extra_is_rejected():
    with pytest.raises((ValueError, ValidationError)):
        ProviderConfig(
            native_structured_output=True,
            extra={"response_format": {"type": "json_object"}},
        )


def test_response_format_in_extra_without_native_is_allowed():
    pc = ProviderConfig(extra={"response_format": {"type": "json_object"}})
    assert pc.native_structured_output is False


# --- build_model_request carries the output schema, system unchanged ---------

def test_build_native_sets_output_format_to_output_class():
    spec = _spec("anthropic/claude-3-5-sonnet-20241022")
    req = build_model_request(spec, In(x=1), model_chain=("m",), native=True)
    assert req.output_format is Out


def test_build_non_native_leaves_output_format_none():
    spec = _spec("anthropic/claude-3-5-sonnet-20241022")
    req = build_model_request(spec, In(x=1), model_chain=("m",), native=False)
    assert req.output_format is None


def test_build_system_is_byte_identical_native_vs_not():
    spec = _spec("anthropic/claude-3-5-sonnet-20241022")
    native = build_model_request(spec, In(x=1), model_chain=("m",), native=True)
    plain = build_model_request(spec, In(x=1), model_chain=("m",), native=False)
    assert native.system == plain.system


# --- resolve-time fail-closed support guard (one-shot only) ------------------

def test_resolve_raises_when_native_unsupported_one_shot(monkeypatch):
    monkeypatch.setattr(
        "drawbore.llm.runtime.litellm.supports_response_schema",
        lambda **kw: False,
    )
    rt = LLMRuntime(config=_native_cfg(), gateway=object(), credential_checker=AllAvailable())
    spec = _spec("profile:judgment")
    with pytest.raises(LLMConfigError):
        rt.resolve(spec)


def test_resolve_passes_when_native_supported_one_shot(monkeypatch):
    seen = []

    def _supports(**kw):
        seen.append(kw)
        return True

    monkeypatch.setattr("drawbore.llm.runtime.litellm.supports_response_schema", _supports)
    rt = LLMRuntime(config=_native_cfg(), gateway=object(), credential_checker=AllAvailable())
    chain = rt.resolve(_spec("profile:judgment"))
    assert chain.attempts[0].provider == "anthropic"
    # exact field mapping: model token AFTER the prefix, provider as custom_llm_provider
    assert seen == [{"model": "claude-3-5-sonnet-20241022", "custom_llm_provider": "anthropic"}]


def test_resolve_does_not_block_loop_agent_even_when_unsupported(monkeypatch):
    called = []
    monkeypatch.setattr(
        "drawbore.llm.runtime.litellm.supports_response_schema",
        lambda **kw: called.append(kw) or False,
    )
    rt = LLMRuntime(config=_native_cfg(), gateway=object(), credential_checker=AllAvailable())
    spec = _spec("profile:judgment", tools=["evidence_retrieve"])
    chain = rt.resolve(spec)  # must NOT raise: native applies to one-shot only
    assert chain.attempts[0].provider == "anthropic"
    assert called == []  # the guard never even probed for a loop agent


def test_resolve_skips_guard_for_providerless_direct_string(monkeypatch):
    called = []
    monkeypatch.setattr(
        "drawbore.llm.runtime.litellm.supports_response_schema",
        lambda **kw: called.append(kw) or False,
    )
    rt = LLMRuntime.from_gateway(object())  # empty config: no providers
    chain = rt.resolve(_spec("gpt-4o"))     # providerless direct string
    assert chain.attempts[0].provider is None
    assert called == []  # providerless attempt → native False → no probe


# --- gateway passes response_format and the reask inherits it ----------------

def _resp(content):
    return {"choices": [{"message": {"content": content}}]}


class _Capturing:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.results[min(len(self.calls) - 1, len(self.results) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


async def test_gateway_passes_response_format_when_output_format_set(monkeypatch):
    fake = _Capturing([_resp(json.dumps({"y": 1}))])
    monkeypatch.setattr("drawbore.llm.production.litellm.acompletion", fake)
    gw = ProductionLLMGateway(config=LLMRuntimeConfig())
    req = ModelRequest(
        system="sys", user="usr",
        model_chain=("anthropic/claude-3-5-sonnet-20241022",), output_format=Out,
    )
    resp = await gw.complete(req)
    assert resp.output == {"y": 1}
    assert fake.calls[0]["response_format"] is Out


async def test_gateway_omits_response_format_when_output_format_none(monkeypatch):
    fake = _Capturing([_resp(json.dumps({"y": 1}))])
    monkeypatch.setattr("drawbore.llm.production.litellm.acompletion", fake)
    gw = ProductionLLMGateway(config=LLMRuntimeConfig())
    req = ModelRequest(system="sys", user="usr", model_chain=("gpt-4o",))
    await gw.complete(req)
    assert "response_format" not in fake.calls[0]


async def test_reask_inherits_response_format(monkeypatch):
    # First call returns a transient empty body (drives one corrective reask), the
    # second returns valid JSON. Both calls must carry response_format.
    fake = _Capturing([_resp(""), _resp(json.dumps({"y": 1}))])
    monkeypatch.setattr("drawbore.llm.production.litellm.acompletion", fake)
    gw = ProductionLLMGateway(config=LLMRuntimeConfig())
    req = ModelRequest(
        system="sys", user="usr",
        model_chain=("anthropic/claude-3-5-sonnet-20241022",), output_format=Out,
    )
    resp = await gw.complete(req)
    assert resp.output == {"y": 1}
    assert len(fake.calls) == 2
    assert fake.calls[0]["response_format"] is Out
    assert fake.calls[1]["response_format"] is Out  # the reask closure inherited it
