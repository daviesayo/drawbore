"""Tool access layer: registry + JIT tokens + proxy."""

from .access import (
    RunContext,
    ToolContext,
    build_tool_context,
    get_run_context,
    reset_run_context,
    set_run_context,
)
from .errors import CircuitBreakerError, TaintError, ToolAccessError, ToolError, TokenError
from .taint import TaintLedger, TrustLabel, join
from .proxy import ToolProxy
from .registry import Tool, ToolRegistry, registry
from .tokens import CapabilityToken, TokenIssuer

__all__ = [
    # registry / execution surface
    "registry",
    "Tool",
    "ToolRegistry",
    "ToolProxy",
    "TokenIssuer",
    "CapabilityToken",
    "ToolContext",
    "build_tool_context",
    "RunContext",
    "set_run_context",
    "get_run_context",
    "reset_run_context",
    # error types (all ToolError subclasses)
    "ToolError",
    "ToolAccessError",
    "TokenError",
    "CircuitBreakerError",
    "TaintError",
    # taint primitives (trust lattice)
    "TrustLabel",
    "TaintLedger",
    "join",
]
