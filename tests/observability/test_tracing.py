import pytest

from opentelemetry.trace import StatusCode

from drawbore.observability import genai_span, get_tracer
from drawbore.observability import semconv


def test_genai_span_emits_a_named_span_with_attributes(captured_spans):
    with genai_span(semconv.OP_EXECUTE_TOOL, "mcp://slack/send", {
        semconv.DRAWBORE_RUN_ID: "r1",
        semconv.GEN_AI_TOOL_NAME: "mcp://slack/send",
        semconv.DRAWBORE_STEP: 0,
    }):
        pass
    spans = captured_spans.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "execute_tool mcp://slack/send"
    assert span.attributes[semconv.GEN_AI_OPERATION_NAME] == "execute_tool"
    assert span.attributes[semconv.DRAWBORE_RUN_ID] == "r1"
    assert span.attributes[semconv.DRAWBORE_STEP] == 0


def test_genai_span_omits_none_valued_attributes(captured_spans):
    with genai_span(semconv.OP_INVOKE_AGENT, "a", {
        semconv.GEN_AI_AGENT_ID: None,          # not set when no identity
        semconv.GEN_AI_AGENT_NAME: "a",
    }):
        pass
    span = captured_spans.get_finished_spans()[0]
    assert semconv.GEN_AI_AGENT_ID not in span.attributes
    assert span.attributes[semconv.GEN_AI_AGENT_NAME] == "a"


def test_genai_span_marks_error_and_records_exception_on_raise(captured_spans):
    with pytest.raises(ValueError):
        with genai_span(semconv.OP_INVOKE_AGENT, "a", {}):
            raise ValueError("boom")
    span = captured_spans.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert any(e.name == "exception" for e in span.events)


def test_get_tracer_returns_a_tracer():
    assert get_tracer() is not None
