"""Register an MCP server's declared tools into a tool registry.

``register_mcp_server`` is a ``drawbore.mcp`` function operating on a registry, so
``drawbore.tools`` never imports ``drawbore.mcp`` (no inverted dependency). Each
declared tool becomes a proxy-backed ``Tool`` (``kind="mcp"``) under
``mcp://<server>/<tool>``; only the declared ``allowed_tools`` are registered —
registering a server does NOT expose its other tools.
"""

from __future__ import annotations

from typing import Any, Iterable

from .client import MCPClient, MCPServerConfig
from .errors import MCPToolNotFoundError


def _make_handler(
    client: MCPClient, session: Any, server_name: str, server_tool_name: str
):
    async def handler(args: Any) -> Any:
        # The proxy calls handler(args) with a single args object; route it to the
        # one MCP tool this handler is bound to. The proxy has already validated the
        # JIT token and tool-level scope before we get here.
        return await client.call_tool(session, server_tool_name, args)

    # Qualify with the server name so two servers exposing the same tool name keep
    # distinct handler identities in tracebacks/traces (the ref already does).
    handler.__name__ = f"mcp::{server_name}/{server_tool_name}"
    return handler


async def register_mcp_server(
    registry,
    *,
    name: str,
    url: str,
    allowed_tools: Iterable[str],
    auth=None,
    client: MCPClient,
) -> tuple[str, ...]:
    """Connect to the MCP server (server-level auth), validate each declared tool
    exists, and register ONLY those tools into ``registry`` as proxy-backed
    ``Tool``s under ``mcp://<name>/<tool>``. Returns the registered tool refs.

    Tool-level scope: a tool the server advertises but the caller did not
    declare is never registered, so an agent can never reach it — even though the
    server would permit it. A declared tool the server does NOT advertise raises
    :class:`MCPToolNotFoundError` (cannot scope to a phantom tool).
    """
    server = MCPServerConfig(name=name, url=url, auth=auth)
    session = await client.connect(server)            # server-level auth happens here
    # On any failure after connect (e.g. a phantom declared tool), close the
    # session so the transport does not leak; the registered handlers keep the
    # session alive only when registration SUCCEEDS.
    try:
        advertised = {spec.name: spec for spec in await client.list_tools(session)}

        refs: list[str] = []
        for tool_name in allowed_tools:
            spec = advertised.get(tool_name)
            if spec is None:
                raise MCPToolNotFoundError(
                    f"MCP server '{name}' does not advertise tool '{tool_name}' "
                    f"(advertised: {sorted(advertised)})"
                )
            ref = f"mcp://{name}/{tool_name}"
            registry.register_mcp_tool(
                ref,
                _make_handler(client, session, name, tool_name),
                schema=spec.input_schema,
            )
            refs.append(ref)
    except BaseException:
        await client.close(session)
        raise
    return tuple(refs)
