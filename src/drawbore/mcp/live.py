"""The live MCP transport — the only `mcp`-SDK user.

Import-guarded: `mcp` is the optional `drawbore[mcp]` extra. Constructing
``LiveMCPClient`` needs nothing; using it without `mcp` installed raises a legible
``MCPError`` telling the user to ``pip install drawbore[mcp]``. A real MCP-server
round-trip is not unit-tested (no live server in CI) — the security model is
proven against ``FakeMCPClient`` (Tasks 3–5).

API note (verified against Context7 ``/modelcontextprotocol/python-sdk``): the SDK
arrives via ``google-adk[mcp]`` (``mcp>=1.24,<2``), which sits between two shapes
the docs describe:

* v1.x: ``from mcp.client.streamable_http import streamablehttp_client``; the
  context manager yields a **3-tuple** ``(read, write, get_session_id)``; OAuth is
  passed directly as ``streamablehttp_client(url, auth=...)``.
* v2 (``main``): ``streamable_http_client``; yields a **2-tuple**
  ``(read, write)``; HTTP/auth is configured on an ``httpx.AsyncClient`` passed as
  ``http_client=``.

Because the test env has no ``mcp`` installed, the exact symbol set shipped by the
pinned ``>=1.24`` build cannot be confirmed empirically in CI. The no-auth connect
path below resolves the transport entry point at runtime (preferring the v2 name,
falling back to the v1 name) and tolerates either tuple arity via ``*_``, so it is
correct against both documented shapes. OAuth wiring is left explicitly
not-yet-wired (see ``_build_oauth_http_client``) rather than guessed across the two
incompatible auth signatures.
"""

from __future__ import annotations

import sys
from typing import Any

from .client import MCPClient, MCPServerConfig, MCPToolSpec
from .errors import MCPError


def _require_mcp():
    # Fail closed with a legible error — never let an opaque ImportError surface.
    # We require the *real* `mcp` SDK, identified by its top-level ``ClientSession``
    # symbol. A bare ``import mcp`` is not sufficient: any module merely named
    # ``mcp`` on the path (e.g. an unrelated package) would import without error,
    # and a later ``from mcp import ClientSession`` would raise an unhelpful
    # ImportError. Checking the sentinel here keeps the failure actionable.
    try:
        from mcp import ClientSession
    except ImportError as exc:  # the extra is not installed (or `mcp` is not the SDK)
        raise MCPError(
            "the MCP transport requires the optional extra: pip install drawbore[mcp]"
        ) from exc
    return ClientSession


def _resolve_streamable_http_client():
    """Return the SDK's Streamable HTTP transport entry point.

    The symbol was renamed between v1.x (``streamablehttp_client``) and v2
    (``streamable_http_client``). Prefer the v2 name; fall back to the v1 name.
    Behind ``_require_mcp`` (the import is guarded).
    """
    from mcp.client import streamable_http as _transport

    for name in ("streamable_http_client", "streamablehttp_client"):
        fn = getattr(_transport, name, None)
        if fn is not None:
            return fn
    raise MCPError(
        "the installed `mcp` SDK exposes neither `streamable_http_client` nor "
        "`streamablehttp_client` in mcp.client.streamable_http; verify the extra"
    )


class LiveMCPClient(MCPClient):
    """Talks to real MCP servers over Streamable HTTP using the `mcp` SDK, with
    server-level OAuth 2.1 / PKCE. Sessions are opaque handles holding the SDK
    transport + ``ClientSession``. Connection pooling / reconnection and full OAuth
    redirect handling are managed-service concerns; this provides a minimal,
    correct no-auth path behind the extra.
    """

    async def connect(self, server: MCPServerConfig) -> Any:
        ClientSession = _require_mcp()

        if server.auth is not None:
            self._reject_oauth_until_wired(server)  # always raises — fail closed

        streamable_http_client = _resolve_streamable_http_client()
        cm = streamable_http_client(server.url)
        # v1.x yields (read, write, get_session_id); v2 yields (read, write).
        read, write, *_ = await cm.__aenter__()
        try:
            # Once the transport cm is open, any failure below must close it —
            # otherwise the socket/HTTP connection leaks (the caller never gets a
            # _LiveSession to close()).
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()
        except BaseException:
            await cm.__aexit__(*sys.exc_info())
            raise
        return _LiveSession(cm=cm, session=session)

    async def list_tools(self, session: Any) -> tuple[MCPToolSpec, ...]:
        result = await session.session.list_tools()
        return tuple(
            MCPToolSpec(
                name=t.name,
                input_schema=getattr(t, "inputSchema", {}) or {},
                description=getattr(t, "description", "") or "",
            )
            for t in result.tools
        )

    async def call_tool(self, session: Any, tool_name: str, arguments: dict) -> Any:
        result = await session.session.call_tool(tool_name, arguments)
        if getattr(result, "isError", False):
            raise MCPError(f"MCP tool '{tool_name}' returned an error: {result}")
        return _unwrap(result)

    async def close(self, session: Any) -> None:
        # Always tear the transport cm down, even if the ClientSession exit raises —
        # otherwise the underlying socket/HTTP connection leaks.
        try:
            await session.session.__aexit__(None, None, None)
        finally:
            await session.cm.__aexit__(None, None, None)

    def _reject_oauth_until_wired(self, server: MCPServerConfig) -> None:
        """Fail closed when server-level OAuth is requested but not yet wired.

        VERIFY-FIRST reconciliation: the SDK's OAuth surface
        (``mcp.client.auth.OAuthClientProvider`` + ``mcp.shared.auth``
        ``OAuthClientMetadata`` / ``TokenStorage``, plus redirect/callback
        handlers) differs between the v1.x ``streamablehttp_client(url, auth=...)``
        call and the v2 ``httpx.AsyncClient(auth=...)`` + ``http_client=`` call.
        Both require token storage and redirect+callback handlers the core does
        not own. The exact shipped (``mcp>=1.24``) signature cannot be confirmed
        without the installed SDK, so per the Implementer note we fail closed
        rather than guess across the two incompatible signatures. Always raises.
        """
        raise MCPError(
            "live OAuth wiring must be completed against the installed mcp SDK "
            "(see the VERIFY-FIRST note); server.auth was provided"
        )


class _LiveSession:
    __slots__ = ("cm", "session")

    def __init__(self, cm, session):
        self.cm = cm
        self.session = session


def _unwrap(result: Any) -> Any:
    """Map an `mcp` CallToolResult to a plain value.

    Verified shape (Context7): a ``CallToolResult`` carries ``structuredContent``
    (a dict, for JSON tools) and/or ``content`` (a list of content blocks, e.g.
    ``TextContent`` with ``.text``). Prefer structured output; otherwise return the
    text of each block.
    """
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    content = getattr(result, "content", None)
    if content:
        return [getattr(block, "text", block) for block in content]
    return None
