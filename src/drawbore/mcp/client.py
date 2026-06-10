"""The MCP transport seam.

``MCPClient`` abstracts the wire protocol so Drawbore's MCP integration — the
security-critical part (server-auth vs tool-scope vs JIT token) — is unit-tested
against ``FakeMCPClient`` with no live server and no `mcp` dependency. The real
``mcp``-SDK client is ``drawbore.mcp.live.LiveMCPClient`` (the `drawbore[mcp]`
extra). A *session* is an opaque handle returned by ``connect``; Drawbore treats
it as a token, never inspects it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .auth import OAuthConfig
from .errors import MCPToolNotFoundError


@dataclass(frozen=True)
class MCPServerConfig:
    """A registered MCP server: a name (the prefix in ``mcp://<name>/<tool>``), a
    URL, and optional server-level OAuth."""

    name: str
    url: str
    auth: OAuthConfig | None = None


@dataclass(frozen=True)
class MCPToolSpec:
    """A tool advertised by an MCP server: its name, JSON-Schema input contract,
    and description. Captured at registration to scope and document the tool."""

    name: str
    input_schema: dict
    description: str = ""


class MCPClient(ABC):
    """Transport for talking to MCP servers. ``connect`` performs server-level
    auth and returns an opaque session handle; ``list_tools`` enumerates the
    server's advertised tools; ``call_tool`` invokes exactly one tool; ``close``
    tears the session down."""

    @abstractmethod
    async def connect(self, server: MCPServerConfig) -> Any:
        """Authenticate (server-level) and return an opaque session handle."""

    @abstractmethod
    async def list_tools(self, session: Any) -> tuple[MCPToolSpec, ...]:
        """Return the tools the server advertises on ``session``."""

    @abstractmethod
    async def call_tool(self, session: Any, tool_name: str, arguments: dict) -> Any:
        """Invoke one tool by name with ``arguments`` and return its result."""

    @abstractmethod
    async def close(self, session: Any) -> None:
        """Tear down ``session``."""


@dataclass(frozen=True, eq=False)
class _FakeSession:
    # ``eq=False`` keeps identity-based ``__hash__``/``__eq__`` so a session can
    # key ``FakeMCPClient.connected_with`` without hashing the server's
    # ``OAuthConfig.extra`` dict (which is unhashable). A session is an opaque
    # handle anyway — Drawbore treats it as a token and never compares by value.
    server: MCPServerConfig


class FakeMCPClient(MCPClient):
    """In-process, dependency-free ``MCPClient`` for tests. Constructed with the
    tools the fake server advertises and the result each returns. Records the auth
    each session connected with and every ``call_tool`` invocation, so tests can
    assert the security separation without a live server."""

    def __init__(self, tools: dict[str, MCPToolSpec], results: dict | None = None):
        self._tools = dict(tools)
        self._results = dict(results or {})
        self.connected_with: dict[Any, MCPServerConfig] = {}
        self.calls: list[tuple[str, dict]] = []
        self.closed: list[Any] = []

    async def connect(self, server: MCPServerConfig) -> Any:
        session = _FakeSession(server=server)
        self.connected_with[session] = server
        return session

    async def list_tools(self, session: Any) -> tuple[MCPToolSpec, ...]:
        return tuple(self._tools.values())

    async def call_tool(self, session: Any, tool_name: str, arguments: dict) -> Any:
        if tool_name not in self._tools:
            raise MCPToolNotFoundError(f"MCP server has no tool '{tool_name}'")
        self.calls.append((tool_name, arguments))
        return self._results.get(tool_name)

    async def close(self, session: Any) -> None:
        self.closed.append(session)
