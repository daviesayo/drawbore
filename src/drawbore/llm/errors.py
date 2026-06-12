"""LLM gateway errors."""

from __future__ import annotations

from drawbore.errors import DrawboreError

# Bound for a surfaced excerpt of a model's offending raw output. Small and fixed:
# enough to diagnose an empty / truncated / non-JSON body, never an unbounded dump.
CONTENT_EXCERPT_LIMIT = 200


def content_excerpt(content: object, *, limit: int = CONTENT_EXCERPT_LIMIT) -> str:
    """A bounded, log-safe excerpt of a model's offending raw output.

    Surfaced in a ``model_error`` reason so a non-JSON / empty / truncated response
    is diagnosable from the halt reason and audit record without dumping unbounded
    model output. The result is the ``repr`` of at most ``limit`` characters (repr
    keeps an empty string, whitespace, and control characters visible and on one
    line), with a trailing marker noting how many characters were clipped. ``None``
    renders as ``<no content>``. This is the model's OWN failed output (not user
    input); it is bounded so it can never become an unbounded log of model content.
    """
    if content is None:
        return "<no content>"
    text = content if isinstance(content, str) else str(content)
    rendered = repr(text[:limit])
    if len(text) > limit:
        rendered += f" (+{len(text) - limit} more chars)"
    return rendered


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
