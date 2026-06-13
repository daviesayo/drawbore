"""Per-run tool access surface and run-context propagation.

``ToolContext`` is handed to an agent that declares tools; it exposes ONLY those
tools as proxy-backed shims, and every call mints a single-use token and routes
through the proxy. The run context flows via a ContextVar set by the pipeline —
never through the agent, so the agent cannot see or mutate run state.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

from .errors import ToolAccessError
from .proxy import ToolProxy
from .tokens import TokenIssuer


@dataclass(frozen=True)
class RunContext:
    run_id: str
    step: int = 0


_current: ContextVar["RunContext | None"] = ContextVar(
    "drawbore_run_context", default=None
)


def set_run_context(ctx: RunContext) -> Token:
    return _current.set(ctx)


def reset_run_context(token: Token) -> None:
    _current.reset(token)


def get_run_context() -> RunContext:
    ctx = _current.get()
    if ctx is None:
        # Typed tool error (not a bare RuntimeError) so callers handling tool
        # access failures can catch it — audit L2.
        raise ToolAccessError("no active run context")
    return ctx


# ---------------------------------------------------------------------------
# Idempotency-key propagation — set by the proxy around each effectful call;
# readable by tool handlers via current_idempotency_key().
# ---------------------------------------------------------------------------

_idempotency_key: ContextVar["str | None"] = ContextVar(
    "drawbore_idempotency_key", default=None
)


def current_idempotency_key() -> "str | None":
    """Return the idempotency key for the currently-executing effectful tool
    call, or ``None`` when called outside an effectful call."""
    return _idempotency_key.get()


def _set_idempotency_key(value: str) -> Token:
    """Internal — set the idempotency key before a handler executes.
    Returns a Token that must be passed to ``_reset_idempotency_key`` after."""
    return _idempotency_key.set(value)


def _reset_idempotency_key(token: Token) -> None:
    """Internal — restore the previous idempotency-key state after a handler."""
    _idempotency_key.reset(token)


ToolShim = Callable[[Any, str], Awaitable[Any]]


class ToolContext:
    """The tool-access object an agent receives. Exposes ONLY its declared tools,
    each as a proxy-backed shim. It holds NO accessible reference to the proxy,
    issuer, or registry — an agent cannot reach a raw handler or mint a token for
    a tool it did not declare (audit C1).

    Residual (accepted, decision-logged): the shims close over the proxy/issuer,
    so a determined in-process attacker could introspect closure cells. True
    cross-agent isolation requires process separation (future infra).
    This object closes the trivial attribute path and removes all access to
    undeclared tools.
    """

    __slots__ = ("__shims",)

    def __init__(self, shims: dict[str, ToolShim]):
        self.__shims = dict(shims)

    async def call(self, tool_ref: str, args: Any, operation: str = "invoke") -> Any:
        shim = self.__shims.get(tool_ref)
        if shim is None:
            raise ToolAccessError(f"agent did not declare tool '{tool_ref}'")
        return await shim(args, operation)


def _make_shim(tool_ref: str, proxy: ToolProxy, issuer: TokenIssuer) -> ToolShim:
    async def shim(args: Any, operation: str = "invoke") -> Any:
        run_ctx = get_run_context()
        token = issuer.issue(tool_ref, run_ctx.run_id, operation)
        return await proxy.invoke(tool_ref, args, token, run_ctx, operation)

    return shim


def build_tool_context(
    declared: Iterable[str], proxy: ToolProxy, issuer: TokenIssuer
) -> ToolContext:
    """Build a ToolContext whose only surface is bound shims for the declared
    tools. The proxy/issuer are captured in closures, never stored as attributes
    (audit C1).
    """
    return ToolContext({ref: _make_shim(ref, proxy, issuer) for ref in declared})
