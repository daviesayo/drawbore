"""MCP native integration. No ADK import; the live `mcp` transport
is behind a guarded import in `drawbore.mcp.live` (the `drawbore[mcp]` extra)."""

from .auth import OAuthConfig
from .client import FakeMCPClient, MCPClient, MCPServerConfig, MCPToolSpec
from .errors import MCPAuthError, MCPError, MCPToolNotFoundError
from .live import LiveMCPClient
from .registration import register_mcp_server

__all__ = [
    "MCPError",
    "MCPAuthError",
    "MCPToolNotFoundError",
    "OAuthConfig",
    "MCPServerConfig",
    "MCPToolSpec",
    "MCPClient",
    "FakeMCPClient",
    "LiveMCPClient",
    "register_mcp_server",
]
