from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.orchestration import LocalEngine
from drawbore.pipeline.executor import StepExecutor
from drawbore.pipeline.outcome import Ok, Halt, StepAudit
from drawbore.tools import TokenIssuer, ToolProxy, registry as default_registry


def test_step_audit_defaults():
    a = StepAudit(input_hash="sha256:x")
    assert a.tool_calls == () and a.model_turns == 0 and a.evidence_summary is None


def test_ok_and_halt_are_distinct():
    a = StepAudit(input_hash=None)
    assert isinstance(Ok(output=None, audit=a), Ok)
    assert isinstance(Halt(reason="r", received=1, attempted=2, audit=a), Halt)


class _In(BaseModel):
    n: int

class _Out(BaseModel):
    n2: int

@agent(name="ex_double", input=_In, output=_Out)
async def _ex_double(v: _In) -> _Out:
    return _Out(n2=v.n * 2)


async def test_executor_returns_ok_for_deterministic_agent():
    issuer = TokenIssuer()
    proxy = ToolProxy(default_registry, issuer)
    ex = StepExecutor(proxy=proxy, issuer=issuer, registry=default_registry,
                      engine=LocalEngine(), evidence_store=None)
    outcome = await ex.execute(
        agent=_ex_double, idx=0, run_id="r", payload={"n": 3},
        evidence_policy=None, tenant_id=None, agent_id=None,
    )
    assert isinstance(outcome, Ok)
    assert outcome.output == _Out(n2=6)
    assert outcome.audit.input_hash is not None


async def test_executor_returns_halt_on_output_violation():
    issuer = TokenIssuer()
    proxy = ToolProxy(default_registry, issuer)

    @agent(name="ex_bad", input=_In, output=_Out)
    async def _bad(v: _In) -> _Out:
        return {"n2": "not-an-int"}

    ex = StepExecutor(proxy=proxy, issuer=issuer, registry=default_registry,
                      engine=LocalEngine(), evidence_store=None)
    outcome = await ex.execute(
        agent=_bad, idx=0, run_id="r", payload={"n": 1},
        evidence_policy=None, tenant_id=None, agent_id=None,
    )
    assert isinstance(outcome, Halt)
    assert outcome.reason.startswith("schema_violation: output:")


async def test_escalation_received_carries_validated_input_on_low_confidence():
    # Regression: after the StepExecutor extraction the scheduler must
    # still pass the step's validated input as the escalation package's `received`
    # at the post-success confidence sites — not None.
    from drawbore.pipeline import Pipeline
    from drawbore.escalation import EscalationPolicy, RecordingDispatcher, HasConfidence

    class _Seed(BaseModel):
        value: int

    class _Score(BaseModel, HasConfidence):
        score: int
        confidence: float

    @agent(name="lowconf_scorer", input=_Seed, output=_Score)
    async def _scorer(v: _Seed) -> _Score:
        return _Score(score=v.value, confidence=0.2)

    d = RecordingDispatcher()
    p = Pipeline(name="t", on_failure=EscalationPolicy("slack", "q"),
                 dispatcher=d, confidence_threshold=0.5)
    p.add(_scorer)
    r = await p.run(_Seed(value=7))

    assert r.status == "escalated"
    assert "confidence_below_threshold" in r.escalations[0].reason
    # The fix: `received` carries the step's validated input, not None.
    assert r.escalations[0].received is not None
    assert r.escalations[0].received == _Seed(value=7)
