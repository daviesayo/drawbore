from drawbore.orchestration.engine import ToolLoopBundle
from drawbore.orchestration.adk_tools import (
    make_loop_before_tool_callback,
    make_loop_model_callbacks,
)
from drawbore.tools import RunContext, TokenIssuer, ToolProxy, ToolRegistry


def _bundle(declared=("echo",)):
    reg = ToolRegistry()
    issuer = TokenIssuer()
    return ToolLoopBundle(
        proxy=ToolProxy(reg, issuer), issuer=issuer, registry=reg,
        declared=declared, run_ctx=RunContext(run_id="r1", step=0),
    )


def test_before_model_counts_turns_and_each_completed_turn_emits_one_closed_span(captured_spans):
    b = _bundle()
    before, after = make_loop_model_callbacks(b, run_id="r1")
    # full before->after round-trip per turn (the real loop calls them in pairs)
    assert before(_FakeCtx(), _FakeReq()) is None     # not skipped
    after(_FakeCtx(), None)
    assert len(b.turns) == 1
    assert before(_FakeCtx(), _FakeReq()) is None
    after(_FakeCtx(), None)
    assert len(b.turns) == 2
    # each completed turn emitted exactly one chat span — and it is CLOSED (finished),
    # proving the span lifecycle does not leak across the two callbacks.
    chat = [s for s in captured_spans.get_finished_spans() if s.name.startswith("chat")]
    assert len(chat) == 2


def test_before_model_short_circuits_once_a_failure_is_recorded(captured_spans):
    b = _bundle()
    b.failures.append(RuntimeError("tool blew up"))
    before, after = make_loop_model_callbacks(b, run_id="r1")
    skip = before(_FakeCtx(), _FakeReq())
    assert skip is not None        # returns a terminating LlmResponse -> model call skipped
    assert len(b.turns) == 0       # no new turn counted after a failure
    # even if ADK still calls after_model on the synthesised response, no span is
    # emitted for a turn that never genuinely started.
    after(_FakeCtx(), None)
    assert [s for s in captured_spans.get_finished_spans() if s.name.startswith("chat")] == []


def test_before_tool_blocks_after_a_failure_and_blocks_undeclared():
    b = _bundle(declared=("echo",))
    cb = make_loop_before_tool_callback(b)
    assert cb(_FakeTool("echo"), {}, None) is None       # declared, no failure -> allowed
    assert isinstance(cb(_FakeTool("secret"), {}, None), dict)   # undeclared -> blocked
    b.failures.append(RuntimeError("x"))
    assert isinstance(cb(_FakeTool("echo"), {}, None), dict)     # after failure -> blocked


class _FakeCtx:
    agent_name = "loop"


class _FakeReq:
    pass


class _FakeTool:
    def __init__(self, name):
        self.name = name
