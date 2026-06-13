"""Tool access layer: registry + JIT tokens + proxy."""

from .access import (
    RunContext,
    ToolContext,
    build_tool_context,
    current_idempotency_key,
    get_run_context,
    reset_run_context,
    set_run_context,
)
from .errors import CircuitBreakerError, TaintError, ToolAccessError, ToolError, TokenError
from .taint import TaintLedger, TrustLabel, join
from .proxy import ToolProxy
from .registry import Tool, ToolRegistry, registry
from .tokens import CapabilityToken, TokenIssuer


def unclassified_effectful_tools(tool_registry: ToolRegistry) -> tuple[str, ...]:
    """Return a sorted tuple of tool refs in ``tool_registry`` currently
    marked ``effectful=True`` (the default), as candidates to review and
    mark ``effectful=False`` if they are pure reads.
    """
    return tuple(
        sorted(
            name
            for name, tool in tool_registry._tools.items()
            if tool.effectful is True
        )
    )


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
    # idempotency-key accessor
    "current_idempotency_key",
    # CI helper
    "unclassified_effectful_tools",
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
