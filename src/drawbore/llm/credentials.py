"""Credential availability seam.

``LLMRuntime`` owns a ``CredentialChecker``. Production reads the named environment
variable; tests inject a fake (``drawbore.testing.StaticCredentialChecker``). The
checker only answers 'is this credential available' — it never reads or returns the
secret VALUE — a credential checker answers availability only, never the value itself.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class CredentialChecker(Protocol):
    """Answers whether a provider credential is available, by env name."""

    def has_credential(self, *, provider: str, credential_env: str | None) -> bool:
        ...


class EnvCredentialChecker:
    """Production default: a credential is available iff its named env var is set to
    a non-empty value. ``credential_env=None`` cannot be confirmed -> ``False``."""

    def has_credential(self, *, provider: str, credential_env: str | None) -> bool:
        if not credential_env:
            return False
        return bool(os.environ.get(credential_env))


class NullCredentialChecker:
    """Always reports credentials as available. Use when the caller supplies its own
    gateway and credential checking is not meaningful (e.g. ``LLMRuntime.from_gateway``).
    Never reads or returns secret values — answers availability only."""

    def has_credential(self, *, provider: str, credential_env: str | None) -> bool:
        return True
