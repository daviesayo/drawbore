"""OTLP export wiring — the `drawbore[otlp]` extra.

The framework is the telemetry source; the customer owns the drain. The OTLP
exporter (`opentelemetry-exporter-otlp`) is an optional extra so the MIT-core base
install stays minimal (Pydantic, ADK, LiteLLM, OpenTelemetry api/sdk).
``configure_otlp_export`` is import-guarded: without the extra it fails closed with
a legible ``ObservabilityError`` (`pip install drawbore[otlp]`).
"""

from __future__ import annotations

from typing import Any

from .errors import ObservabilityError


def _import_otlp_span_exporter():
    """Return an OTLP span-exporter class from the `otlp` extra. Prefer the HTTP
    exporter; fall back to gRPC. Raise a legible ``ObservabilityError`` if neither
    is installed (the extra is absent)."""
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        return OTLPSpanExporter
    except ImportError:
        pass
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        return OTLPSpanExporter
    except ImportError as exc:
        raise ObservabilityError(
            "OTLP export requires the optional extra: pip install drawbore[otlp]"
        ) from exc


def configure_otlp_export(
    endpoint: str,
    *,
    headers: dict[str, str] | None = None,
    set_global: bool = True,
) -> Any:
    """Build a ``TracerProvider`` that batches Drawbore's spans to an OTLP drain
    (Axiom/Datadog/Grafana/Honeycomb/any OTLP backend) at ``endpoint``. Sets it as
    the OTel global provider when ``set_global`` (the default). Returns the
    provider. Requires the `drawbore[otlp]` extra — fails closed otherwise.
    """
    # Check the optional exporter FIRST so the realistic failure (the `otlp` extra
    # is absent) fails closed with the legible ObservabilityError before anything
    # else runs — mirrors the `_require_mcp()`-first guard idiom.
    exporter_cls = _import_otlp_span_exporter()

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    exporter = (
        exporter_cls(endpoint=endpoint, headers=headers)
        if headers is not None
        else exporter_cls(endpoint=endpoint)
    )
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    if set_global:
        trace.set_tracer_provider(provider)
    return provider
