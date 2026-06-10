import sys

import pytest

from drawbore.observability import ObservabilityError, configure_otlp_export


def test_configure_otlp_export_fails_closed_without_the_extra(monkeypatch):
    # Deterministically simulate the extra being absent: setting a module to None
    # in sys.modules makes `from <that module> import X` raise ImportError. We hide
    # BOTH the http and grpc exporter modules so the guard's fallback is exhausted.
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None)
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.grpc.trace_exporter", None)
    with pytest.raises(ObservabilityError) as ei:
        configure_otlp_export("http://localhost:4318")
    assert "drawbore[otlp]" in str(ei.value)


def test_configure_otlp_export_builds_a_provider_when_available():
    pytest.importorskip("opentelemetry.exporter.otlp")
    provider = configure_otlp_export("http://localhost:4318", set_global=False)
    # A real provider that yields a tracer; no global mutation, no network.
    assert provider.get_tracer("t") is not None
