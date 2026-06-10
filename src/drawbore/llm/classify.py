"""Classify provider/LiteLLM exceptions into the closed fallback taxonomy.

A broad ``except Exception: try next model`` is not acceptable: auth/config errors
must surface as ``model_config_error`` and contract errors as ``model_error``, never
silent fallback. Only ``kind == "transport"`` with a ``reason`` in the profile's
``fallback_on`` may advance to the next provider.

litellm is imported here, which is allowed: ``classify`` lives under ``drawbore.llm``,
the only subsystem permitted to import litellm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import litellm.exceptions as _e

from .config import FallbackReason


@dataclass(frozen=True)
class ProviderOutcome:
    """How a provider exception should be treated. ``kind``:
    ``transport`` -> may fall back if ``reason`` is in ``fallback_on``;
    ``auth`` -> ``model_config_error`` (no fallback);
    ``contract`` -> ``model_error`` (no fallback);
    ``unknown`` -> fail closed without fallback."""

    kind: Literal["transport", "auth", "contract", "unknown"]
    reason: FallbackReason | None = None


# Ordered most-specific FIRST. In litellm 1.87.0, ``Timeout`` subclasses
# ``APIConnectionError`` (Timeout -> APITimeoutError -> APIConnectionError -> APIError),
# so ``Timeout`` MUST be matched before ``APIConnectionError`` or it would be swallowed
# as ``provider_unavailable`` instead of ``timeout``. ``InternalServerError`` and the
# other status classes are independent ``APIStatusError`` subclasses (no shadowing among
# them), but the broad ``APIError`` base is deliberately NOT listed: an unrecognised
# ``APIError`` subtype falls through to ``unknown`` (fail closed, never silent fallback).
_TRANSPORT: tuple[tuple[type, FallbackReason], ...] = (
    (_e.Timeout, "timeout"),
    (_e.RateLimitError, "rate_limit"),
    (_e.ServiceUnavailableError, "provider_unavailable"),
    (_e.APIConnectionError, "provider_unavailable"),
    (_e.InternalServerError, "server_error"),
)
# Auth/config faults -> model_config_error, never fallback. No ordering constraints
# (independent APIStatusError siblings); matched by isinstance-any.
_AUTH: tuple[type, ...] = (_e.AuthenticationError, _e.PermissionDeniedError)
# Contract faults (the model/request is wrong) -> model_error, never fallback. Add
# further contract types (e.g. UnprocessableEntityError) here as needed.
_CONTRACT: tuple[type, ...] = (_e.BadRequestError,)


def classify_provider_exception(exc: BaseException) -> ProviderOutcome:
    """Map a raised provider exception to a :class:`ProviderOutcome`.

    Matching is by ``isinstance`` against the curated tuples in most-specific order.
    Anything not explicitly recognised returns ``kind="unknown"`` so the caller fails
    closed rather than treating an unknown error as a fallback opportunity."""
    for typ, reason in _TRANSPORT:
        if isinstance(exc, typ):
            return ProviderOutcome(kind="transport", reason=reason)
    if isinstance(exc, _AUTH):
        return ProviderOutcome(kind="auth")
    if isinstance(exc, _CONTRACT):
        return ProviderOutcome(kind="contract")
    return ProviderOutcome(kind="unknown")
