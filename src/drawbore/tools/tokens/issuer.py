"""Single-use capability tokens for tool access.

A CapabilityToken is an in-process, opaque, NON-serializable capability scoped to
exactly one tool, one operation, and one pipeline run, with a hard expiry. The
agent never holds a persistent credential; the proxy consumes the token on use.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Callable

from ..errors import TokenError


class CapabilityToken:
    """Opaque, single-use, non-serializable capability. The secret is never
    exposed via ``repr`` and the object refuses to be pickled."""

    __slots__ = ("_secret", "tool_ref", "operation", "run_id")

    def __init__(self, secret: str, tool_ref: str, operation: str, run_id: str):
        self._secret = secret
        self.tool_ref = tool_ref
        self.operation = operation
        self.run_id = run_id

    def __repr__(self) -> str:
        return (
            f"CapabilityToken(tool_ref={self.tool_ref!r}, "
            f"operation={self.operation!r}, run_id={self.run_id!r})"
        )

    def __reduce__(self):
        raise TypeError("CapabilityToken is not serializable")

    def __getstate__(self):
        raise TypeError("CapabilityToken is not serializable")


@dataclass
class _Record:
    tool_ref: str
    operation: str
    run_id: str
    expires_at: float
    used: bool = False


class TokenIssuer:
    """Issues and consumes single-use capability tokens. ``clock`` returns
    monotonic seconds and is injectable for testing expiry."""

    def __init__(self, ttl_seconds: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self._ttl = ttl_seconds
        self._clock = clock
        self._records: dict[str, _Record] = {}

    def issue(self, tool_ref: str, run_id: str, operation: str = "invoke") -> CapabilityToken:
        secret = secrets.token_urlsafe(16)
        self._records[secret] = _Record(
            tool_ref=tool_ref,
            operation=operation,
            run_id=run_id,
            expires_at=self._clock() + self._ttl,
        )
        return CapabilityToken(secret, tool_ref, operation, run_id)

    def consume(
        self, token: CapabilityToken, tool_ref: str, run_id: str, operation: str = "invoke"
    ) -> None:
        rec = self._records.get(token._secret)
        if rec is None:
            raise TokenError("unknown capability token")
        if rec.used:
            raise TokenError("capability token already used (single-use)")
        if rec.tool_ref != tool_ref or rec.run_id != run_id or rec.operation != operation:
            raise TokenError("capability token scope mismatch")
        if self._clock() > rec.expires_at:
            raise TokenError("capability token expired")
        rec.used = True
