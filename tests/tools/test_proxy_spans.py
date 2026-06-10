import pytest

from opentelemetry.trace import StatusCode

from drawbore.observability import semconv
from drawbore.tools import (
    ToolRegistry, ToolProxy, TokenIssuer, RunContext, CircuitBreakerError,
)


async def _ok(args):
    return {"echoed": args}


async def _wire(max_calls=3):
    registry = ToolRegistry()
    registry.register_tool("greet", _ok)
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer, max_calls_per_tool=max_calls)
    return registry, issuer, proxy


async def test_successful_tool_call_emits_an_execute_tool_span(captured_spans):
    _, issuer, proxy = await _wire()
    rc = RunContext(run_id="r1", step=0)
    token = issuer.issue("greet", "r1", "invoke")
    await proxy.invoke("greet", {"name": "x"}, token, rc)
    spans = captured_spans.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "execute_tool greet"
    assert span.attributes[semconv.GEN_AI_OPERATION_NAME] == "execute_tool"
    assert span.attributes[semconv.GEN_AI_TOOL_NAME] == "greet"
    assert span.attributes[semconv.DRAWBORE_RUN_ID] == "r1"
    assert span.attributes[semconv.DRAWBORE_STEP] == 0
    assert span.attributes[semconv.DRAWBORE_STATUS] == "ok"
    assert semconv.DRAWBORE_INPUT_HASH in span.attributes
    assert semconv.DRAWBORE_OUTPUT_HASH in span.attributes


async def test_circuit_broken_call_emits_an_error_span_with_denial_label(captured_spans):
    _, issuer, proxy = await _wire(max_calls=1)
    rc = RunContext(run_id="r1", step=0)
    await proxy.invoke("greet", {}, issuer.issue("greet", "r1", "invoke"), rc)  # 1st ok
    with pytest.raises(CircuitBreakerError):
        await proxy.invoke("greet", {}, issuer.issue("greet", "r1", "invoke"), rc)  # 2nd trips
    spans = captured_spans.get_finished_spans()
    assert len(spans) == 2
    broken = spans[1]
    assert broken.status.status_code == StatusCode.ERROR
    assert broken.attributes[semconv.DRAWBORE_STATUS] == "denied:breaker"


async def test_span_hash_matches_payload_hash(captured_spans):
    from drawbore.observability import payload_hash
    _, issuer, proxy = await _wire()
    rc = RunContext(run_id="r1", step=0)
    await proxy.invoke("greet", {"name": "x"}, issuer.issue("greet", "r1", "invoke"), rc)
    span = captured_spans.get_finished_spans()[0]
    assert span.attributes[semconv.DRAWBORE_INPUT_HASH] == payload_hash({"name": "x"})
