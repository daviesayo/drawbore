"""Model resolution: profile grammar + chain resolver.

Centralizes profile expansion so the gateway and the ADK loop never reimplement it.
``drawbore.llm`` stays a near-leaf; this imports only stdlib + the config/errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .config import FallbackReason, LLMRuntimeConfig
from .credentials import CredentialChecker
from .errors import LLMConfigError

PROFILE_PREFIX = "profile:"
_PROFILE_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")


def is_profile_ref(value: str) -> bool:
    """True if ``value`` claims the reserved ``profile:`` prefix (it may still be
    malformed — that is reported by :func:`parse_profile_ref`)."""
    return value.startswith(PROFILE_PREFIX)


def parse_profile_ref(value: str) -> str:
    """Return the profile ``<name>`` for a ``profile:<name>`` ref, or raise
    :class:`LLMConfigError` (``model_config_error``) for any malformed ref:
    empty name, whitespace, nested prefix, or invalid characters."""
    if not is_profile_ref(value):
        raise LLMConfigError(f"not a profile ref: {value!r}")
    name = value[len(PROFILE_PREFIX):]
    if not _PROFILE_NAME_RE.match(name):
        raise LLMConfigError(
            f"malformed profile reference {value!r}: name must match "
            f"[A-Za-z][A-Za-z0-9_.-]{{0,63}} with no whitespace or nested prefix"
        )
    return name


@dataclass(frozen=True)
class ModelAttempt:
    """One concrete provider attempt in a resolved chain. ``request_model``
    is the provider-prefixed string a gateway calls; ``provider`` is ``None`` only for
    a providerless direct string."""

    provider: str | None
    model: str
    request_model: str
    declared_ref: str
    source: Literal["profile", "direct"]
    fallback_on: tuple[FallbackReason, ...]
    credential_env: str | None
    credential_required: bool


@dataclass(frozen=True)
class ResolvedModelChain:
    """The ordered, de-duplicated attempts for a model-backed agent."""

    declared: tuple[str, ...]
    attempts: tuple[ModelAttempt, ...]


def _request_model(provider: str | None, model: str) -> str:
    return f"{provider}/{model}" if provider else model


def _attempts_for_ref(
    ref: str, config: LLMRuntimeConfig, *, credential_checker: CredentialChecker
) -> list[ModelAttempt]:
    """Expand one declared ref (profile or direct) to its ordered attempts, checking
    credentials. Raises ``LLMConfigError`` for missing profile / missing provider
    credential_env / missing required credential / malformed ref."""
    if is_profile_ref(ref):
        name = parse_profile_ref(ref)                       # raises on malformed
        profile = config.profiles.get(name)
        if profile is None:
            raise LLMConfigError(f"profile {name!r} is not configured (ref {ref!r})")
        out: list[ModelAttempt] = []
        for target in profile.targets:
            provider_cfg = config.providers.get(target.provider)
            credential_env = provider_cfg.credential_env if provider_cfg else None
            if target.credential_required:
                if credential_env is None:
                    raise LLMConfigError(
                        f"profile {name!r} target {target.provider}/{target.model} "
                        f"requires a credential but provider {target.provider!r} has "
                        f"no ProviderConfig.credential_env configured"
                    )
                if not credential_checker.has_credential(
                    provider=target.provider, credential_env=credential_env
                ):
                    raise LLMConfigError(
                        f"missing required credential for provider {target.provider!r} "
                        f"(env {credential_env}) in profile {name!r}"
                    )
            out.append(ModelAttempt(
                provider=target.provider, model=target.model,
                request_model=_request_model(target.provider, target.model),
                declared_ref=ref, source="profile", fallback_on=profile.fallback_on,
                credential_env=credential_env, credential_required=target.credential_required,
            ))
        return out

    # Direct string. Any ``profile:``-prefixed ref was already routed into the profile
    # branch above (is_profile_ref), where parse_profile_ref rejects malformed names,
    # so a string reaching here is never profile-prefixed.
    head, sep, tail = ref.partition("/")
    if sep and head and tail:
        provider, model = head, tail
    else:
        provider, model = None, ref
    return [ModelAttempt(
        provider=provider, model=model, request_model=_request_model(provider, model),
        declared_ref=ref, source="direct",
        fallback_on=("timeout", "rate_limit", "server_error", "provider_unavailable"),
        credential_env=None, credential_required=False,  # direct strings cannot be preflighted
    )]


def resolve_chain(
    spec, config: LLMRuntimeConfig, *, credential_checker: CredentialChecker
) -> ResolvedModelChain:
    """Resolve ``spec.model`` then ``spec.fallback_model`` into a de-duplicated
    ``ResolvedModelChain``. ``spec`` is an ``AgentSpec`` (typed loosely to
    keep ``llm`` from importing ``agent``). Raises ``LLMConfigError`` on any setup
    failure; NEVER calls a provider. Dedup identity: ``(provider, model)``, or
    ``(None, request_model)`` for providerless direct strings."""
    declared = tuple(r for r in (spec.model, spec.fallback_model) if r)
    if not declared:
        raise LLMConfigError(
            f"agent '{spec.name}' is model-backed but declares no model"
        )
    attempts: list[ModelAttempt] = []
    seen: set[tuple] = set()
    for ref in declared:
        for attempt in _attempts_for_ref(ref, config, credential_checker=credential_checker):
            key = (attempt.provider, attempt.model) if attempt.provider else (None, attempt.request_model)
            if key in seen:
                continue
            seen.add(key)
            attempts.append(attempt)
    if not attempts:
        raise LLMConfigError(f"agent '{spec.name}' resolved to an empty model chain")
    return ResolvedModelChain(declared=declared, attempts=tuple(attempts))
