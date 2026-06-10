"""Opt-in confidence marker for escalation."""

from __future__ import annotations


class HasConfidence:
    """Marker mixin. Inherit this on an agent OUTPUT model to opt into
    confidence-based escalation; the model MUST itself declare a float
    ``confidence`` field in [0, 1].

    Only models that explicitly inherit this marker are checked against the
    pipeline's confidence threshold — an incidental field named ``confidence`` on a
    model that does NOT inherit the marker is ignored. The framework never
    fabricates a confidence value.
    """
