"""Observability — OTel GenAI-semconv span emission. A leaf module:
no ADK, no mcp, no imports from agent/pipeline/tools/llm. The OTLP exporter ships
behind the `drawbore[otlp]` extra (see `export.py`)."""

from .errors import ObservabilityError
from .export import configure_otlp_export
from .hashing import payload_hash
from .tracing import (
    genai_span,
    get_tracer,
    reset_tracer_provider,
    set_attributes,
    use_tracer_provider,
)
from . import semconv

__all__ = [
    "ObservabilityError",
    "configure_otlp_export",
    "payload_hash",
    "genai_span",
    "get_tracer",
    "set_attributes",
    "use_tracer_provider",
    "reset_tracer_provider",
    "semconv",
]
