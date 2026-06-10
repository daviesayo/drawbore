"""Orchestration errors. Self-declares ``halt_reason`` so an engine-misconfiguration
failure escalates legibly without ``drawbore.errors`` importing
``drawbore.orchestration``."""

from __future__ import annotations

from drawbore.errors import DrawboreError


class EngineError(DrawboreError):
    """An orchestration-engine contract violation — e.g. a model-backed agent run on
    an engine that has no model path. Engine-agnostic: the message names no concrete
    engine and never names ADK (principle 6)."""

    halt_reason = "engine_error"
