"""Tool call proxy — the single chokepoint for every tool call."""

from .proxy import DEFAULT_MAX_CALLS_PER_TOOL, ToolProxy

__all__ = ["DEFAULT_MAX_CALLS_PER_TOOL", "ToolProxy"]
