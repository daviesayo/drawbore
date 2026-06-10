import pytest
from pydantic import ValidationError

from drawbore.llm import (
    LLMRuntimeConfig,
    ModelProfile,
    ModelTarget,
    ProviderConfig,
)


def test_model_target_holds_provider_and_model():
    t = ModelTarget(provider="openrouter", model="anthropic/claude-3-5-sonnet")
    assert t.provider == "openrouter"
    assert t.model == "anthropic/claude-3-5-sonnet"
    assert t.credential_required is True


def test_profile_canonicalizes_provider_prefixed_string_target():
    p = ModelProfile(targets=("openrouter/anthropic/claude-3-5-sonnet",))
    assert p.targets == (ModelTarget(provider="openrouter", model="anthropic/claude-3-5-sonnet"),)


def test_profile_rejects_providerless_string_target():
    with pytest.raises(ValidationError):
        ModelProfile(targets=("claude-3-5-sonnet",))


def test_profile_rejects_empty_targets():
    with pytest.raises(ValidationError):
        ModelProfile(targets=())


def test_profile_default_fallback_reasons():
    p = ModelProfile(targets=(ModelTarget(provider="openai", model="gpt-4o"),))
    assert p.fallback_on == ("timeout", "rate_limit", "server_error", "provider_unavailable")


def test_configs_forbid_extra_keys():
    with pytest.raises(ValidationError):
        LLMRuntimeConfig(profiles={}, providers={}, bogus=1)
    with pytest.raises(ValidationError):
        ModelTarget(provider="x", model="y", bogus=1)
    with pytest.raises(ValidationError):
        ProviderConfig(credential_env="K", bogus=1)


def test_provider_config_extra_is_not_shared_mutable_default():
    a = ProviderConfig()
    b = ProviderConfig()
    a.extra["k"] = 1
    assert b.extra == {}  # distinct dict instances, no shared mutable default


def test_runtime_config_defaults_are_empty_and_distinct():
    a = LLMRuntimeConfig()
    b = LLMRuntimeConfig()
    a.profiles["p"] = ModelProfile(targets=(ModelTarget(provider="openai", model="gpt-4o"),))
    assert b.profiles == {}
