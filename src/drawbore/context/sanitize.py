"""External-input sanitisation — typed boundary + structural bounds.

Sanitisation is NOT content scrubbing. External input is bounded (size and depth)
and (by the caller) parsed into a declared schema; it is always treated as data,
never as instructions. The architectural basis: string-scrubbing prompt injection
is unreliable — the real defence is architectural (data never becomes instructions).
"""

from __future__ import annotations

from typing import Any

from drawbore.errors import SanitizationError

DEFAULT_MAX_BYTES = 1_048_576
DEFAULT_MAX_DEPTH = 32


def sanitize(
    value: Any,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> Any:
    """Enforce structural bounds on an external value and return it unchanged.

    Raises :class:`SanitizationError` if the value nests deeper than ``max_depth``
    or its representation exceeds ``max_bytes``.

    Intended for JSON-like external input (dicts, lists, scalars); sets/frozensets
    are not traversed for depth.
    """
    _check_depth(value, max_depth, 1)
    size = len(repr(value).encode("utf-8"))
    if size > max_bytes:
        raise SanitizationError(f"external input size {size} exceeds limit {max_bytes}")
    return value


def _check_depth(value: Any, max_depth: int, depth: int) -> None:
    if depth > max_depth:
        raise SanitizationError(f"external input nesting exceeds depth limit {max_depth}")
    if isinstance(value, dict):
        for v in value.values():
            _check_depth(v, max_depth, depth + 1)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _check_depth(v, max_depth, depth + 1)
