"""Typed runtime LLM config. Policy, not provider clients.

Closed by default (``extra="forbid"``), except the deliberate ``ProviderConfig.extra``
pass-through for provider-specific LiteLLM/OpenRouter parameters. This config is
SEPARATE from pipeline JSON: a manifest may carry ``"model": "profile:judgment"``
as an agent declaration string, but never credentials, gateways, engines, or routing
knobs. ``drawbore.llm`` stays a near-leaf — stdlib + pydantic + drawbore.errors only.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import TypeAliasType

# A closed fallback taxonomy. Mapped by provider adapters; never free-form.
FallbackReason = Literal[
    "timeout", "rate_limit", "server_error", "provider_unavailable", "schema_violation"
]

# Provider-specific pass-through values (the one open door, ProviderConfig.extra). A named
# recursive alias (Python 3.11 has no PEP 695 `type`); TypeAliasType lets Pydantic build the
# recursive schema without RecursionError. typing_extensions is a Pydantic transitive dep.
JsonValue = TypeAliasType(
    "JsonValue",
    "str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]",
)

_DEFAULT_FALLBACK: tuple[FallbackReason, ...] = (
    "timeout", "rate_limit", "server_error", "provider_unavailable",
)


class ModelTarget(BaseModel):
    """Binds a concrete model id to a provider key. ``credential_required``
    gates the runtime credential preflight for this target."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    credential_required: bool = True

    @classmethod
    def from_string(cls, raw: str) -> "ModelTarget":
        """Canonicalize a provider-prefixed string target such as
        ``"openrouter/anthropic/claude-3-5-sonnet"`` to
        ``ModelTarget(provider="openrouter", model="anthropic/claude-3-5-sonnet")``.
        A providerless string (no ``/``) is INVALID inside a profile (profiles are
        production policy and must bind each model to a provider key)."""
        head, sep, tail = raw.partition("/")
        if not sep or not head or not tail:
            raise ValueError(
                f"profile string target {raw!r} must be provider-prefixed "
                f"(e.g. 'openrouter/anthropic/claude-3-5-sonnet'); providerless "
                f"string targets are not allowed in a profile"
            )
        return cls(provider=head, model=tail)


class ModelProfile(BaseModel):
    """A named profile: an ordered, non-empty target list + a closed fallback set."""

    model_config = ConfigDict(extra="forbid")

    targets: tuple[ModelTarget, ...]
    fallback_on: tuple[FallbackReason, ...] = _DEFAULT_FALLBACK

    @field_validator("targets", mode="before")
    @classmethod
    def _canonicalize_targets(cls, value):
        # Accept ModelTarget | provider-prefixed str; canonicalize strings.
        if value is None:
            return value
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(ModelTarget.from_string(item))
            else:
                out.append(item)
        return tuple(out)

    @model_validator(mode="after")
    def _require_targets(self):
        if not self.targets:
            raise ValueError("a ModelProfile must declare at least one target")
        return self


class ProviderConfig(BaseModel):
    """Optional provider-specific runtime metadata. References credentials by env
    name; never stores secret values. ``extra`` is the one explicit
    pass-through door for provider params Drawbore does not model."""

    model_config = ConfigDict(extra="forbid")

    credential_env: str | None = None
    base_url: str | None = None
    timeout_seconds: float | None = None
    native_structured_output: bool = False
    extra: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_native_response_format_collision(self):
        # Native mode auto-wires each agent's output schema as ``response_format``; a
        # ``response_format`` key in ``extra`` would be the only thing that could collide.
        # Reject both-set at construction (fail-closed) instead of silently dropping one.
        if self.native_structured_output and "response_format" in self.extra:
            raise ValueError(
                "native_structured_output=True cannot be combined with a "
                "'response_format' key in extra; native mode owns response_format"
            )
        return self


class LLMRuntimeConfig(BaseModel):
    """The deployment's model policy: named profiles + per-provider config."""

    model_config = ConfigDict(extra="forbid")

    profiles: dict[str, ModelProfile] = Field(default_factory=dict)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
