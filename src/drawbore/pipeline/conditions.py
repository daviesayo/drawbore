"""Serializable branch conditions. Leaf operators only — no combinators.

A `When` gates a node: the node runs iff the condition is true against produced
outputs. It addresses an upstream field as "agent.field" (like `From`), serializes
to the JSON config manifest, and renders to plain English for the audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import BaseModel

_OPS = ("equals", "in_", "is_true", "gt", "lt")


@dataclass(frozen=True)
class When:
    ref: str                                  # "agent.field"
    equals: Any | None = None  # NOTE: None cannot be used as an equals value; None is the "operator not set" sentinel
    in_: tuple[Any, ...] | None = None
    is_true: bool | None = None
    gt: float | None = None
    lt: float | None = None

    def __post_init__(self) -> None:
        if "." not in self.ref:
            raise ValueError(
                f"When.ref must be 'agent.field' (gate on a field, not a whole object); got {self.ref!r}"
            )
        set_ops = [op for op in _OPS if getattr(self, op) is not None]
        if len(set_ops) != 1:
            raise ValueError(
                f"When requires exactly one operator from {_OPS}; got {set_ops or 'none'}"
            )

    @property
    def agent(self) -> str:
        return self.ref.split(".", 1)[0]

    @property
    def field(self) -> str:
        return self.ref.split(".", 1)[1]

    def evaluate(self, outputs: Mapping[str, BaseModel]) -> bool:
        value = getattr(outputs[self.agent], self.field)
        if self.equals is not None:
            return value == self.equals
        if self.in_ is not None:
            return value in self.in_
        if self.is_true is not None:
            return bool(value) is self.is_true
        if self.gt is not None:
            return value > self.gt
        return value < self.lt           # lt is the only remaining operator

    def legible(self) -> str:
        if self.equals is not None:
            return f"{self.ref} == {self.equals!r}"
        if self.in_ is not None:
            return f"{self.ref} in {list(self.in_)}"
        if self.is_true is not None:
            return f"{self.ref} is {'true' if self.is_true else 'false'}"
        if self.gt is not None:
            return f"{self.ref} > {self.gt}"
        return f"{self.ref} < {self.lt}"
