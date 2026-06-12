"""The explicit, Drawbore-owned model request/response.

These are deliberately framework types — not ADK or LiteLLM internals — so the
model boundary stays inspectable and a future pre-model transform can read and
rewrite a request without reaching into any engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from .attempts import ModelAudit  # acyclic: attempts imports only stdlib
from .usage import TokenUsage  # acyclic: usage imports only stdlib


@dataclass(frozen=True)
class ModelRequest:
    """A single non-streaming model call. ``system`` carries the agent's
    instructions plus the JSON output contract; ``user`` is the validated input
    rendered as data; ``model_chain`` is the resolved primary-then-fallback chain.
    """

    system: str
    user: str
    model_chain: tuple[str, ...]


@dataclass(frozen=True)
class ModelResponse:
    """The parsed result of a model call. ``output`` is the JSON the model
    produced, parsed to a ``dict`` (the pipeline validates it against the agent's
    output model). ``raw_text`` is retained for audit/debugging; ``audit``
    carries the provider-attempt summary when produced by the runtime
    (``None`` on the gateway-only path). ``usage`` is the call's token counts and
    ``cost`` the provider-reported cost — each ``None`` when the provider does not
    report it (never fabricated)."""

    output: dict
    model_used: str
    raw_text: str
    audit: ModelAudit | None = None
    usage: TokenUsage | None = None
    cost: float | None = None
