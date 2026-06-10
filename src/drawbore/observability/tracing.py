"""OTel tracing helpers — the single seam through which Drawbore emits spans.

``genai_span`` is a context manager that opens a span named ``"{operation}
{target}"`` (per the GenAI semconv span-naming rule), stamps
``gen_ai.operation.name`` + the given attributes (skipping ``None`` values), and
on an exception marks the span ERROR + records the exception before re-raising.

Tracer resolution goes through an overridable module-level provider so tests can
capture spans in-process without mutating OTel global state (the ``captured_spans``
fixture uses ``use_tracer_provider`` / ``reset_tracer_provider``). In production,
leave the override unset and either set the OTel global provider yourself or call
``configure_otlp_export``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from .semconv import GEN_AI_OPERATION_NAME

_TRACER_NAME = "drawbore"
_provider_override: Any = None


def use_tracer_provider(provider: Any) -> Any:
    """Override the provider Drawbore emits through (tests / explicit in-process
    wiring). Returns the previous override so it can be restored."""
    global _provider_override
    previous = _provider_override
    _provider_override = provider
    return previous


def reset_tracer_provider(previous: Any) -> None:
    """Restore the provider override returned by :func:`use_tracer_provider`."""
    global _provider_override
    _provider_override = previous


def get_tracer():
    """Return Drawbore's tracer — from the override if set, else the OTel global
    provider (a no-op tracer until the host configures one)."""
    if _provider_override is not None:
        return _provider_override.get_tracer(_TRACER_NAME)
    return trace.get_tracer(_TRACER_NAME)


def set_attributes(span: Span, attributes: dict | None) -> None:
    """Set every non-``None`` attribute on ``span`` (None means 'not applicable',
    e.g. an absent agent id, and is omitted rather than stringified)."""
    if not attributes:
        return
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


@contextmanager
def genai_span(operation: str, target: str, attributes: dict | None = None) -> Iterator[Span]:
    """Open a GenAI-semconv span ``"{operation} {target}"``; mark ERROR + record
    the exception on raise (then re-raise). The caller may set further attributes
    or an explicit OK status on the yielded span."""
    tracer = get_tracer()
    with tracer.start_as_current_span(f"{operation} {target}") as span:
        span.set_attribute(GEN_AI_OPERATION_NAME, operation)
        set_attributes(span, attributes)
        try:
            yield span
        except BaseException as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise
