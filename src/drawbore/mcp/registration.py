"""Register an MCP server's declared tools into a tool registry.

``register_mcp_server`` is a ``drawbore.mcp`` function operating on a registry, so
``drawbore.tools`` never imports ``drawbore.mcp`` (no inverted dependency). Each
declared tool becomes a proxy-backed ``Tool`` (``kind="mcp"``) under
``mcp://<server>/<tool>``; only the declared ``allowed_tools`` are registered —
registering a server does NOT expose its other tools.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable, Mapping

from .client import MCPClient, MCPServerConfig
from .errors import MCPToolNotFoundError

if TYPE_CHECKING:
    from ..tools.taint import TrustLabel


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


async def _connect_and_register(
    registry,
    *,
    name: str,
    url: str,
    allowed_tools: Iterable[str],
    auth=None,
    client: MCPClient,
    source_trust: "Mapping[str, TrustLabel] | None" = None,
    exfil_capable: Mapping[str, bool] | None = None,
) -> tuple[tuple[str, ...], Any]:
    """Connect to the MCP server, validate declared tools exist, and register them.

    Returns ``(refs, session)``. Flag validation happens before connecting (no
    cleanup needed on error there). On any failure after connect the session is
    closed and the exception re-raised so the transport never leaks.
    """
    declared = tuple(allowed_tools)
    declared_set = set(declared)
    source_trust_map = dict(source_trust or {})
    exfil_map = dict(exfil_capable or {})
    for flag_name, mapping in (("source_trust", source_trust_map), ("exfil_capable", exfil_map)):
        phantom = set(mapping) - declared_set
        if phantom:
            raise ValueError(
                f"{flag_name} declares tool(s) not in allowed_tools for MCP server "
                f"'{name}': {sorted(phantom)} (allowed: {sorted(declared_set)})"
            )

    server = MCPServerConfig(name=name, url=url, auth=auth)
    session = await client.connect(server)            # server-level auth happens here
    # On any failure after connect (e.g. a phantom declared tool), close the
    # session so the transport does not leak; the registered handlers keep the
    # session alive only when registration SUCCEEDS.
    try:
        advertised = {spec.name: spec for spec in await client.list_tools(session)}

        refs: list[str] = []
        for tool_name in declared:
            spec = advertised.get(tool_name)
            if spec is None:
                raise MCPToolNotFoundError(
                    f"MCP server '{name}' does not advertise tool '{tool_name}' "
                    f"(advertised: {sorted(advertised)})"
                )
            ref = f"mcp://{name}/{tool_name}"
            # Default fail-safe: register_mcp_tool's own defaults (UNTRUSTED source,
            # not exfil-capable) apply unless the caller declared otherwise. Pass a
            # flag only when present so the registry default is the single source of
            # the fail-safe.
            extra: dict[str, Any] = {}
            if tool_name in source_trust_map:
                extra["source_trust"] = source_trust_map[tool_name]
            if tool_name in exfil_map:
                extra["exfil_capable"] = exfil_map[tool_name]
            registry.register_mcp_tool(
                ref,
                _make_handler(client, session, name, tool_name),
                schema=spec.input_schema,
                **extra,
            )
            refs.append(ref)
    except BaseException:
        await client.close(session)
        raise
    return tuple(refs), session


async def register_mcp_server(
    registry,
    *,
    name: str,
    url: str,
    allowed_tools: Iterable[str],
    auth=None,
    client: MCPClient,
    source_trust: "Mapping[str, TrustLabel] | None" = None,
    exfil_capable: Mapping[str, bool] | None = None,
) -> tuple[str, ...]:
    """Connect to the MCP server (server-level auth), validate each declared tool
    exists, and register ONLY those tools into ``registry`` as proxy-backed
    ``Tool``s under ``mcp://<name>/<tool>``. Returns the registered tool refs.

    Tool-level scope: a tool the server advertises but the caller did not
    declare is never registered, so an agent can never reach it — even though the
    server would permit it. A declared tool the server does NOT advertise raises
    :class:`MCPToolNotFoundError` (cannot scope to a phantom tool).

    Per-tool trust declarations: ``source_trust`` and ``exfil_capable`` are
    optional maps keyed by tool name (a name in ``allowed_tools``). They let an MCP
    tool participate in the taint gate with the same honesty as a custom tool — an
    outward-writing MCP tool (a notifier, an email relay) declared
    ``exfil_capable={"notify": True}`` is refused while the step's data is
    untrusted. A tool absent from a map keeps the fail-safe MCP default:
    untrusted-source and not exfil-capable. A flag keyed to a tool that is not in
    ``allowed_tools`` raises :class:`ValueError` (cannot scope a flag to a tool the
    agent can never reach), before the server is contacted.
    """
    refs, _session = await _connect_and_register(
        registry,
        name=name,
        url=url,
        allowed_tools=allowed_tools,
        auth=auth,
        client=client,
        source_trust=source_trust,
        exfil_capable=exfil_capable,
    )
    return refs


@asynccontextmanager
async def mcp_server(
    registry,
    *,
    name: str,
    url: str,
    allowed_tools: Iterable[str],
    auth=None,
    client: MCPClient,
    source_trust: "Mapping[str, TrustLabel] | None" = None,
    exfil_capable: Mapping[str, bool] | None = None,
) -> AsyncIterator[tuple[str, ...]]:
    """Async context manager that registers an MCP server's declared tools and
    guarantees teardown of the underlying session on exit.

    Takes the same parameters as :func:`register_mcp_server` and yields the same
    tuple of registered refs. The session opened during registration is closed in
    a ``finally`` block so the transport never leaks, even when the caller raises
    inside the block.

    Idempotent exit: calling ``__aexit__`` more than once is safe — the
    ``asynccontextmanager`` protocol exhausts the generator on the first exit,
    making subsequent exits no-ops.

    Use this when the server connection should be scoped to a block of work.
    Use :func:`register_mcp_server` directly when you manage the session
    lifecycle yourself.
    """
    refs, session = await _connect_and_register(
        registry,
        name=name,
        url=url,
        allowed_tools=allowed_tools,
        auth=auth,
        client=client,
        source_trust=source_trust,
        exfil_capable=exfil_capable,
    )
    try:
        yield refs
    finally:
        await client.close(session)
