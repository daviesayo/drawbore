"""Explicit Join nodes. A Join merges branches by a combine policy:

- exactly_one      — exactly one source ran (else halt); forward it.
- first_by_priority — forward the first present source in `sources` order (halt if zero).
- all_present       — every source ran (else halt); build `output` from `inputs` bindings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Literal, get_args

from pydantic import BaseModel

if TYPE_CHECKING:
    from .binding import From

JoinPolicy = Literal["exactly_one", "first_by_priority", "all_present"]


@dataclass
class JoinNode:
    name: str
    sources: list[str]
    policy: JoinPolicy
    output: type[BaseModel]
    inputs: dict[str, "From"] = field(default_factory=dict)  # only for all_present (From bindings)


def Join(
    name: str,
    *,
    sources: Iterable[str],
    policy: JoinPolicy,
    output: type[BaseModel],
    inputs: dict[str, "From"] | None = None,
) -> JoinNode:
    if policy not in get_args(JoinPolicy):
        raise ValueError(f"unknown join policy {policy!r}")
    if not sources:
        raise ValueError("Join requires at least one source")
    return JoinNode(name=name, sources=list(sources), policy=policy,
                    output=output, inputs=dict(inputs or {}))


class JoinError(Exception):
    """Raised by the evaluator on a policy violation (scheduler maps to a halt)."""
