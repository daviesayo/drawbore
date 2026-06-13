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
    """A circuit-breaker cap was exceeded, halting the call before it reached the handler.

    Covers two distinct breaker levels, both raising this error and logging
    ``denied:breaker``:

    - **Per-step / per-tool** (``max_calls_per_tool``): an agent called the same
      tool more than the allowed number of times within a single pipeline step.
    - **Run-level total** (``max_tool_calls_per_run``): the total number of
      admitted tool calls across all steps of the run exceeded the run cap.
    - **Run-level distinct** (``max_distinct_tools_per_run``): the run attempted
      to invoke more distinct tool refs than the run cap allows.

    The halt code in all cases is ``circuit_breaker`` (``denied:breaker`` in the
    proxy log). Distinguishing per-step from run-level breaches requires reading
    the legible error message attached to the exception.
    """


# A ToolError so the proxy denial path and model-loop abort handle it uniformly;
# intentionally not a subclass of the registered halt types, so halt_reason_for
# classifies it via the halt_reason attribute.
class TaintError(ToolError):
    """An exfil-capable tool was invoked while the step's taint scope was UNTRUSTED (the lethal-trifecta breaker)."""

    halt_reason = "taint_violation"
