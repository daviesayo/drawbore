import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.llm import (
    EnvCredentialChecker,
    LLMConfigError,
    LLMRuntimeConfig,
    ModelProfile,
    ModelTarget,
    ProviderConfig,
)
from drawbore.llm.resolution import ResolvedModelChain, resolve_chain


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


def _spec(model, fallback_model=None):
    @agent(name="a", input=In, output=Out, model=model, fallback_model=fallback_model)
    async def a(v: In) -> Out: ...
    return a.spec


class AllAvailable:
    def has_credential(self, *, provider, credential_env):
        return True


def _cfg():
    return LLMRuntimeConfig(
        profiles={
            "judgment": ModelProfile(targets=(
                ModelTarget(provider="openrouter", model="anthropic/claude-3-5-sonnet"),
                ModelTarget(provider="openai", model="gpt-4o"),
            )),
        },
        providers={
            "openrouter": ProviderConfig(credential_env="OPENROUTER_API_KEY"),
            "openai": ProviderConfig(credential_env="OPENAI_API_KEY"),
        },
    )


def test_profile_expands_to_ordered_chain():
    chain = resolve_chain(_spec("profile:judgment"), _cfg(), credential_checker=AllAvailable())
    assert isinstance(chain, ResolvedModelChain)
    assert chain.declared == ("profile:judgment",)
    assert [(a.provider, a.model) for a in chain.attempts] == [
        ("openrouter", "anthropic/claude-3-5-sonnet"),
        ("openai", "gpt-4o"),
    ]
    assert chain.attempts[0].request_model == "openrouter/anthropic/claude-3-5-sonnet"
    assert chain.attempts[0].source == "profile"
    assert chain.attempts[0].credential_env == "OPENROUTER_API_KEY"


def test_direct_provider_prefixed_string_passes_through():
    chain = resolve_chain(
        _spec("openrouter/anthropic/claude-3-5-sonnet"),
        LLMRuntimeConfig(), credential_checker=AllAvailable(),
    )
    a = chain.attempts[0]
    assert a.source == "direct"
    assert a.provider == "openrouter"
    assert a.model == "anthropic/claude-3-5-sonnet"
    assert a.request_model == "openrouter/anthropic/claude-3-5-sonnet"
    assert a.credential_required is False  # direct strings cannot be preflighted


def test_providerless_direct_string_has_no_provider():
    chain = resolve_chain(_spec("gpt-4o"), LLMRuntimeConfig(), credential_checker=AllAvailable())
    a = chain.attempts[0]
    assert a.provider is None
    assert a.request_model == "gpt-4o"


def test_model_then_fallback_dedup_by_provider_model():
    chain = resolve_chain(
        _spec("profile:judgment", "openrouter/anthropic/claude-3-5-sonnet"),
        _cfg(), credential_checker=AllAvailable(),
    )
    # the fallback's (openrouter, anthropic/claude-3-5-sonnet) duplicates profile attempt 0
    keys = [(a.provider, a.model) for a in chain.attempts]
    assert keys == [("openrouter", "anthropic/claude-3-5-sonnet"), ("openai", "gpt-4o")]
    assert chain.declared == ("profile:judgment", "openrouter/anthropic/claude-3-5-sonnet")


def test_missing_profile_raises_config_error():
    with pytest.raises(LLMConfigError):
        resolve_chain(_spec("profile:nope"), _cfg(), credential_checker=AllAvailable())


def test_no_declared_model_raises_config_error():
    # A spec with neither model nor fallback has no chain to resolve; resolve_chain
    # fails closed with a legible config error rather than producing an empty chain.
    with pytest.raises(LLMConfigError):
        resolve_chain(_spec(None), LLMRuntimeConfig(), credential_checker=AllAvailable())


def test_missing_required_credential_raises_config_error():
    class NoneAvailable:
        def has_credential(self, *, provider, credential_env):
            return False
    with pytest.raises(LLMConfigError):
        resolve_chain(_spec("profile:judgment"), _cfg(), credential_checker=NoneAvailable())


def test_profile_target_with_no_providers_entry_raises():
    cfg = LLMRuntimeConfig(profiles={
        "p": ModelProfile(targets=(ModelTarget(provider="openai", model="gpt-4o"),)),
    })  # no providers[] entry -> credential_required target has no credential_env
    with pytest.raises(LLMConfigError):
        resolve_chain(_spec("profile:p"), cfg, credential_checker=AllAvailable())


def test_malformed_profile_ref_raises_config_error():
    with pytest.raises(LLMConfigError):
        resolve_chain(_spec("profile:"), _cfg(), credential_checker=AllAvailable())
