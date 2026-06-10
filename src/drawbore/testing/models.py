"""Mock value resolution for local test mode.

A tool mock may be: a static value, a sync callable ``(args) -> value``, an async
callable ``async (args) -> value``, or a sequence of values/exceptions consumed in
order. ``make_tool_handler`` turns any of these into a uniform async handler the
scoped registry installs. Exceptions in a sequence are RAISED (they flow through
``ToolProxy.invoke`` so the proxy records the failure and the pipeline halts)."""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Sequence, Union

ToolMock = Union[Any, Callable[[Any], Any], Sequence[Any]]
# Keyed by agent name; values are output dicts/models/sequences/callables(ModelRequest).
ModelMock = Any
# Keyed by agent name; a sequence of LoopScript turns (see loop.py).
LoopScript = Any


def _is_exception(value: Any) -> bool:
    return isinstance(value, BaseException) or (
        isinstance(value, type) and issubclass(value, BaseException)
    )


def _raise(value: Any) -> None:
    raise value() if isinstance(value, type) else value


async def _resolve_one(value: Any, args: Any) -> Any:
    """Resolve a single (non-sequence) mock value against call ``args``."""
    if _is_exception(value):
        _raise(value)
    if callable(value):
        result = value(args)
        if inspect.isawaitable(result):
            return await result
        return result
    return value


def make_tool_handler(ref: str, mock: ToolMock) -> Callable[[Any], Awaitable[Any]]:
    """Build an async handler for a tool mock. A list/tuple is a sequence consumed in
    order (exhaustion -> ``TestingError``); anything else is resolved per call."""
    from .errors import TestingError

    if isinstance(mock, (list, tuple)):
        remaining = list(mock)

        async def seq_handler(args: Any) -> Any:
            if not remaining:
                raise TestingError(f"mock tool '{ref}' sequence is exhausted")
            return await _resolve_one(remaining.pop(0), args)

        return seq_handler

    async def handler(args: Any) -> Any:
        return await _resolve_one(mock, args)

    return handler
