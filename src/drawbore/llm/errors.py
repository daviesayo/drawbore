"""LLM gateway errors."""

from __future__ import annotations

from drawbore.errors import DrawboreError


class LLMError(DrawboreError):
    """Base class for model-boundary errors.

    Declares ``halt_reason`` so ``drawbore.errors.halt_reason_for`` classifies a
    model-boundary failure legibly without ``drawbore.errors`` importing
    ``drawbore.llm`` (which would be a cycle). ``"model_error"`` covers a model
    returning content Drawbore cannot use (non-JSON / malformed-shape / non-object
    200); subclasses override for a more specific reason.
    """

    halt_reason = "model_error"


class ModelUnavailableError(LLMError):
    """Every model in the resolved fallback chain failed. The pipeline
    halts-and-escalates with the legible reason ``"model_unavailable"`` so an
    escalation reader can tell the LLM provider chain was exhausted, not that the
    agent's own code threw."""

    halt_reason = "model_unavailable"


class LLMConfigError(LLMError):
    """Bad runtime LLM setup: missing profile, empty chain, malformed profile
    reference, missing required credential, invalid provider option, unsupported
    runtime config. Distinct from ``model_unavailable`` (provider attempts failed)
    and ``model_error`` (model contract failure) so an escalation reader can tell
    'we configured this wrong' from 'OpenRouter timed out'."""

    halt_reason = "model_config_error"
