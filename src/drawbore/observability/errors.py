"""Observability/export configuration errors."""

from __future__ import annotations

from drawbore.errors import DrawboreError


class ObservabilityError(DrawboreError):
    """A telemetry *configuration* failure — e.g. ``configure_otlp_export`` called
    without the ``drawbore[otlp]`` extra installed. This is a setup-time error
    (fail closed), not a run-time halt, so it declares no
    ``halt_reason``."""
