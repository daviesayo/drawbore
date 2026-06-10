"""The ``@agent`` decorator."""

from __future__ import annotations

from typing import Callable, Literal

from pydantic import BaseModel

from .spec import AgentFn, AgentSpec


class Agent:
    """A deterministic Drawbore agent: a callable carrying its frozen
    :class:`AgentSpec`. An agent function has the signature
    ``async def fn(value: InputModel) -> OutputModel``. Agents that declare
    tools have the signature ``async def fn(value: InputModel, tools: ToolContext) -> OutputModel``.
    """

    def __init__(self, spec: AgentSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    async def __call__(self, value: BaseModel) -> BaseModel:
        return await self.spec.fn(value)


def agent(
    *,
    input: type[BaseModel],
    output: type[BaseModel],
    name: str | None = None,
    tools: list[str] | None = None,
    context_access: Literal["none"] = "none",
    requires_human_approval: bool = False,
    risk_tier: Literal["low", "medium", "high", "critical"] = "low",
    version: str = "0.0.0",
    model: str | None = None,
    fallback_model: str | None = None,
    instructions: str | None = None,
) -> Callable[[AgentFn], Agent]:
    """Decorate an async function into an :class:`Agent`.

    The developer writes business logic only; the framework owns schema
    enforcement at the boundaries.
    """

    def decorate(fn: AgentFn) -> Agent:
        spec = AgentSpec(
            name=name or fn.__name__,
            input=input,
            output=output,
            fn=fn,
            context_access=context_access,
            tools=tuple(tools or ()),
            requires_human_approval=requires_human_approval,
            risk_tier=risk_tier,
            version=version,
            model=model,
            fallback_model=fallback_model,
            instructions=instructions,
        )
        return Agent(spec)

    return decorate
