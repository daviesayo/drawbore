"""Bidirectional agent ref resolution.

Import needs ``ref -> Agent``; export needs ``Agent -> ref``. ``AgentCatalog`` is
the explicit, deterministic path. ``resolve_ref``/``ref_for`` accept EITHER an
``AgentCatalog`` or a plain ``Mapping[str, Agent]`` — a mapping is inverted by
object identity and fails closed unless exactly one ref points at the agent (zero =
uncataloged, multiple = ambiguous)."""

from __future__ import annotations

from typing import Mapping

from drawbore.agent import Agent

from .errors import ConfigResolutionError


class AgentCatalog:
    """An explicit, bidirectional ``ref <-> Agent`` registry.

    ``ref_for`` keys by object identity (``id(agent)``) — the catalog holds each
    agent alive via ``_by_ref``, so the id stays stable for the catalog's lifetime.
    """

    def __init__(self) -> None:
        self._by_ref: dict[str, Agent] = {}
        self._ref_by_id: dict[int, str] = {}

    def register(self, ref: str, agent: Agent) -> None:
        """Register a ``ref -> Agent`` binding. Fails closed if the same agent
        is already bound to a different ref (export must have exactly one ref per
        agent), or if the ref is already bound to a different agent (a ref must
        resolve unambiguously). Re-registering the identical ``(ref, agent)`` pair is
        an idempotent no-op."""
        existing_ref = self._ref_by_id.get(id(agent))
        if existing_ref is not None and existing_ref != ref:
            raise ConfigResolutionError(
                f"agent '{agent.name}' is already registered under ref '{existing_ref}'; "
                f"cannot also register it as '{ref}' (exactly one ref per agent)"
            )
        existing_agent = self._by_ref.get(ref)
        if existing_agent is not None and existing_agent is not agent:
            raise ConfigResolutionError(
                f"ref '{ref}' is already registered to a different agent "
                f"('{existing_agent.name}'); a ref must map to exactly one agent"
            )
        self._by_ref[ref] = agent
        self._ref_by_id[id(agent)] = ref

    def resolve(self, ref: str) -> Agent:
        try:
            return self._by_ref[ref]
        except KeyError:
            raise ConfigResolutionError(f"agent ref '{ref}' is not registered") from None

    def ref_for(self, agent: Agent) -> str:
        # register() guarantees one-ref-per-agent, so the only failure here is a
        # genuinely uncataloged agent — name that case, don't imply ambiguity.
        ref = self._ref_by_id.get(id(agent))
        if ref is None:
            raise ConfigResolutionError(
                f"agent '{agent.name}' is not registered in the catalog"
            )
        return ref


def resolve_ref(agents: "AgentCatalog | Mapping[str, Agent]", ref: str) -> Agent:
    """``ref -> Agent`` for import, against a catalog or a plain mapping."""
    if isinstance(agents, AgentCatalog):
        return agents.resolve(ref)
    try:
        return agents[ref]
    except KeyError:
        raise ConfigResolutionError(f"agent ref '{ref}' is not registered") from None


def ref_for(agents: "AgentCatalog | Mapping[str, Agent]", agent: Agent) -> str:
    """``Agent -> ref`` for export. A mapping is inverted by object identity and
    fails closed on zero (uncataloged) or multiple (ambiguous) matches."""
    if isinstance(agents, AgentCatalog):
        return agents.ref_for(agent)
    matches = [ref for ref, a in agents.items() if a is agent]
    if not matches:
        raise ConfigResolutionError(
            f"agent '{agent.name}' is not in the agent mapping (uncataloged)"
        )
    if len(matches) > 1:
        raise ConfigResolutionError(
            f"agent '{agent.name}' has no unique catalog ref (ambiguous: {sorted(matches)})"
        )
    return matches[0]
