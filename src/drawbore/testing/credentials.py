"""Fake credential checker for local test mode.

Default: every credential is 'available' so CI does not need real provider keys for
mocked profile attempts. Pass ``available=False`` to prove missing-credential
behavior (the run halts with ``model_config_error``)."""

from __future__ import annotations


class StaticCredentialChecker:
    """Reports a fixed availability for every credential."""

    def __init__(self, *, available: bool = True) -> None:
        self._available = available

    def has_credential(self, *, provider: str, credential_env: str | None) -> bool:
        return self._available
