"""MCP integration errors.

These self-declare ``halt_reason`` so a failed MCP tool call escalates with a
legible reason (``"mcp_error"``) rather than the generic ``"agent_error"``,
keeping ``drawbore.errors`` free of any ``drawbore.mcp`` import.
"""

from __future__ import annotations

from drawbore.errors import DrawboreError


class MCPError(DrawboreError):
    """Base class for MCP integration failures (registration, transport, tool call).

    Declares ``halt_reason`` so ``drawbore.errors.halt_reason_for`` classifies an
    MCP failure as ``"mcp_error"`` without ``drawbore.errors`` importing
    ``drawbore.mcp`` (which would be a cycle). Subclasses inherit the reason
    unless they override it.
    """

    halt_reason = "mcp_error"


class MCPAuthError(MCPError):
    """Server-level authentication failed at registration (OAuth 2.1 / PKCE)."""


class MCPToolNotFoundError(MCPError):
    """A declared ``allowed_tool`` is not advertised by the MCP server at
    registration time. Registration halts — the server cannot be scoped to a tool
    it does not expose."""
