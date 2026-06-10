"""Server-level MCP authentication config.

``OAuthConfig`` is the registration-time, server-level credential (OAuth 2.1 with
PKCE, MCP's native auth standard). It is consumed once at ``MCPClient.connect`` and
NEVER reaches the agent or the per-invocation token path — server auth and
tool-level scope are different layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OAuthConfig:
    """OAuth 2.1 client config for server-level auth. PKCE is the assumed flow;
    no client secret lives in the core (the live transport supplies token storage
    and redirect handling). ``scopes`` are server-level grants, distinct from the
    per-tool scope the proxy enforces."""

    client_id: str
    scopes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)
