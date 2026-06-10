"""Resolve an agent's model fallback chain."""

from __future__ import annotations

from .errors import LLMError


def resolve_model_chain(spec) -> tuple[str, ...]:
    """Build the ordered, de-duplicated model chain for a model-backed agent: its
    ``model`` first, then its ``fallback_model``. Raises :class:`LLMError` if the
    spec declares no model (a model-less agent is deterministic and never reaches
    here). ``spec`` is a ``drawbore.agent.AgentSpec`` (typed loosely to keep ``llm``
    from importing ``agent``).

    A gateway/engine-level *default* model is intentionally not added: an
    agent is model-backed iff it declares a ``model`` (no dead default path). A
    default can be layered on later if a real use case needs it.
    """
    ordered: list[str] = []
    for candidate in (spec.model, spec.fallback_model):
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    if not ordered:
        raise LLMError(
            f"agent '{spec.name}' is model-backed but declares no model "
            f"(set @agent(model=...))"
        )
    return tuple(ordered)
