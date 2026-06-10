# Test configuration. asyncio_mode=auto (pyproject) makes async test functions
# run without an explicit marker.

import pytest


@pytest.fixture
def captured_spans():
    """Capture Drawbore's OTel spans in-process for assertions.

    Installs an in-memory span exporter on a fresh ``TracerProvider``, points
    ``drawbore.observability`` at it via the provider-override seam, yields the
    exporter (call ``.get_finished_spans()``), and restores the previous override
    afterwards. No OTLP extra, no network, no OTel global-provider mutation — so
    tests stay isolated from each other and from process-wide OTel state.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from drawbore import observability

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous = observability.use_tracer_provider(provider)
    try:
        yield exporter
    finally:
        observability.reset_tracer_provider(previous)
        exporter.clear()


# A fake ADK BaseLlm returning SCRIPTED turns (no network, no live model).
# Lives here (not a test-package module) so both tests/orchestration and
# tests/pipeline can use it without a cross-package import (there is no
# tests/__init__.py). google-adk is a core dep, so this top-level import is safe.
from google.adk.models import BaseLlm, LlmResponse  # noqa: E402
from google.genai import types  # noqa: E402


class FakeADKModel(BaseLlm):
    def __init__(self, script):
        super().__init__(model="fake-model")
        # BaseLlm is a pydantic model — keep non-field state off the pydantic surface.
        object.__setattr__(self, "_script", list(script))
        object.__setattr__(self, "_i", 0)

    async def generate_content_async(self, llm_request, stream=False):
        i = self._i
        object.__setattr__(self, "_i", i + 1)
        if i >= len(self._script):
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="{}")]))
            return
        turn = self._script[i]
        kind = turn[0]
        if kind == "call":
            _, name, args = turn
            yield LlmResponse(content=types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
            ))
        elif kind == "multicall":
            _, calls = turn
            parts = [types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in calls]
            yield LlmResponse(content=types.Content(role="model", parts=parts))
        else:
            _, text = turn
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


@pytest.fixture
def fake_adk_model():
    """Return a factory ``make(script) -> FakeADKModel`` for the agentic-loop tests."""
    def _make(script):
        return FakeADKModel(script)
    return _make
