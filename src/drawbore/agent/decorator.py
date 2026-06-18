"""The ``@agent`` decorator."""

from __future__ import annotations

from typing import Callable, Literal, Sequence

from pydantic import BaseModel

from .spec import AgentFn, AgentSpec


def _compose_instructions(value: str | Sequence[str] | None) -> str | None:
    """Normalise the ``instructions`` argument to a single string (or ``None``).

    A plain ``str`` (or ``None``) is returned unchanged, so existing behaviour is
    byte-identical. A sequence of fragments is composed in order: each fragment is
    stripped, blank fragments are dropped, and the rest are joined with a blank
    line. An empty or all-blank sequence normalises to ``None``. The result is
    stored in the frozen ``AgentSpec``, so nothing downstream — ``build.py``, the
    manifest round-trip — sees anything but a string.

    Composition is **static**: fragments are fixed at decoration time. Instructions
    are never derived from a step's input, which would let untrusted data steer the
    system prompt (input is data, not authority).
    """
    if value is None or isinstance(value, str):
        return value
    fragments = [str(fragment).strip() for fragment in value]
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments:
        return None
    return "\n\n".join(fragments)


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
    instructions: str | Sequence[str] | None = None,
) -> Callable[[AgentFn], Agent]:
    """Decorate an async function into an :class:`Agent`.

    The developer writes business logic only; the framework owns schema
    enforcement at the boundaries.

    ``instructions`` may be a single string or a sequence of fragments composed
    in order (shared preamble + agent body + standing constraints) — see
    :func:`_compose_instructions`. Composition is static; instructions are never
    derived from a step's input.
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
            instructions=_compose_instructions(instructions),
        )
        return Agent(spec)

    return decorate
