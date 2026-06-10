"""The frozen specification carried by every Drawbore agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Literal

from pydantic import BaseModel

AgentFn = Callable[[BaseModel], Awaitable[BaseModel]]


@dataclass(frozen=True)
class AgentSpec:
    """Immutable declaration of an agent's contract.

    ``context_access`` is ``Literal["none"]``: an agent sees only its
    orchestrator-constructed input — no ambient, shared, or run state.
    """

    name: str
    input: type[BaseModel]
    output: type[BaseModel]
    fn: AgentFn
    context_access: Literal["none"] = "none"
    tools: tuple[str, ...] = ()
    requires_human_approval: bool = False
    risk_tier: Literal["low", "medium", "high", "critical"] = "low"
    version: str = "0.0.0"
    model: str | None = None
    fallback_model: str | None = None
    instructions: str | None = None
