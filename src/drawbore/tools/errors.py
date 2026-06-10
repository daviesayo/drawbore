"""Tool access layer errors."""

from __future__ import annotations


class ToolError(Exception):
    """Base class for tool access layer failures."""


class ToolAccessError(ToolError):
    """An agent attempted to use a tool it did not declare, an unknown tool, or a
    disallowed operation. Also a security event."""


class TokenError(ToolError):
    """A capability token was unknown, already used, out of scope, or expired."""


class CircuitBreakerError(ToolError):
    """An agent exceeded the allowed number of calls to a tool within one run."""


# A ToolError so the proxy denial path and model-loop abort handle it uniformly;
# intentionally not a subclass of the registered halt types, so halt_reason_for
# classifies it via the halt_reason attribute.
class TaintError(ToolError):
    """An exfil-capable tool was invoked while the step's taint scope was UNTRUSTED (the lethal-trifecta breaker)."""

    halt_reason = "taint_violation"
