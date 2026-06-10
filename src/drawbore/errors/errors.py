"""Drawbore error hierarchy and halt classification.

The framework default is halt-and-escalate, not degrade: any runtime failure
stops the pipeline and is reported with a legible reason. ``halt_reason_for``
maps a caught exception to that reason.
"""

from __future__ import annotations

from drawbore.schema.errors import SchemaValidationError
from drawbore.tools.errors import CircuitBreakerError, TokenError, ToolAccessError


class DrawboreError(Exception):
    """Base class for Drawbore framework errors."""


class SanitizationError(DrawboreError):
    """External input exceeded a structural bound (size or depth)."""


class ResumeDriftError(DrawboreError):
    """A checkpointed run was resumed under drifted pipeline semantics.

    The pipeline refuses such a resume before any step executes; this error
    type exists for taxonomy completeness and for callers that want to raise
    the condition themselves.
    """

    halt_reason = "resume_drift"


# Most-specific first. A subclass MUST appear before any base class it inherits
# from, otherwise the base entry would shadow it in halt_reason_for. The ordering
# invariant is enforced at import time by the guard below.
_HALT_REASONS: tuple[tuple[type, str], ...] = (
    (SchemaValidationError, "schema_violation"),
    (CircuitBreakerError, "circuit_breaker"),
    (TokenError, "token_violation"),
    (ToolAccessError, "tool_access"),
    (SanitizationError, "sanitization"),
)

# Invariant (enforced at import): a more-specific subclass must precede any base
# class it derives from, otherwise the base would shadow it in halt_reason_for.
# Today the listed types are unrelated siblings, so this is a guard against future
# edits, not a current necessity.
for _i, (_typ, _) in enumerate(_HALT_REASONS):
    for _later_typ, _ in _HALT_REASONS[_i + 1:]:
        assert not issubclass(_later_typ, _typ), (
            f"{_later_typ.__name__} must be listed before its base {_typ.__name__}"
        )


def halt_reason_for(exc: BaseException) -> str:
    """Classify a caught exception into a legible halt reason.

    Precedence: the curated ``_HALT_REASONS`` registry (authoritative, ordering-
    guarded) is consulted first; then a self-declared ``halt_reason`` class
    attribute on the exception; then ``"agent_error"``. The attribute is the
    extension hook for errors defined in modules that ``drawbore.errors`` cannot
    import without a cycle (e.g. ``drawbore.llm``, whose errors import
    ``DrawboreError`` from here) — they declare their own legible reason instead of
    being registered here.

    Unknown, non-declaring exceptions are ``"agent_error"`` — the pipeline still
    halts (fail closed); it never continues on an unclassified failure.
    """
    for typ, reason in _HALT_REASONS:
        if isinstance(exc, typ):
            return reason
    declared = getattr(exc, "halt_reason", None)
    if isinstance(declared, str) and declared:
        return declared
    return "agent_error"


# The closed, documented halt-code vocabulary. Every halted RunResult carries
# exactly one of these in `halt_code`. Sources: the curated registry above, the
# self-declared `halt_reason` class attributes (drawbore.llm/.evidence/.config/
# .mcp/.testing/.orchestration/.tools errors), and the pipeline's own
# control-flow halts. Extending this set is an interface change: update the
# reliability guide in the same commit.
HALT_CODES: tuple[str, ...] = (
    "schema_violation",
    "circuit_breaker",
    "token_violation",
    "tool_access",
    "sanitization",
    "taint_violation",
    "model_error",
    "model_unavailable",
    "model_config_error",
    "evidence_error",
    "config_resolution_error",
    "authority_regression",
    "mcp_error",
    "testing_error",
    "engine_error",
    "agent_error",
    "join_policy_violation",
    "condition_unevaluable",
    "condition_tainted",
    "identity_blocked",
    "confidence_marker_without_value",
    "confidence_below_threshold",
    "requires_human_approval",
    "resume_drift",
)
