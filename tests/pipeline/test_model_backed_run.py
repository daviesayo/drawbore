"""End-to-end proof: a model-backed agent runs through the pipeline with every
Drawbore guarantee intact.

The pipeline is model-agnostic: ``ADKEngine`` returns the model's structured
output as a raw ``dict`` that the pipeline validates against the agent's output
model exactly like any other raw output. No pipeline change is required.
"""

from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.pipeline import Pipeline, From
from drawbore.orchestration import ADKEngine
from drawbore.llm import LLMGateway, ModelResponse, ModelUnavailableError
from drawbore.escalation import EscalationPolicy, RecordingDispatcher


class Seed(BaseModel):
    amount: int


class Score(BaseModel):
    risk: str


class _Gateway(LLMGateway):
    def __init__(self, output):
        self._output = output

    async def complete(self, request):
        return ModelResponse(output=self._output, model_used="m1", raw_text="")


class _FailingGateway(LLMGateway):
    async def complete(self, request):
        raise ModelUnavailableError("down")


async def test_model_backed_agent_runs_end_to_end_and_is_validated():
    @agent(name="scorer", input=Seed, output=Score, model="m1", instructions="score")
    async def scorer(v: Seed) -> Score:
        raise AssertionError("fn must not run for a model-backed agent")

    p = Pipeline(name="t")
    p.add(scorer)
    r = await p.run(Seed(amount=100), engine=ADKEngine(gateway=_Gateway({"risk": "high"})))
    assert r.status == "completed"
    assert isinstance(r.outputs["scorer"], Score)  # the dict was validated to the model
    assert r.outputs["scorer"].risk == "high"


async def test_model_output_violating_schema_halts():
    @agent(name="scorer", input=Seed, output=Score, model="m1")
    async def scorer(v: Seed) -> Score:
        raise AssertionError

    p = Pipeline(name="t")
    p.add(scorer)
    # gateway returns a dict missing the required 'risk' field → boundary validation fails
    r = await p.run(Seed(amount=1), engine=ADKEngine(gateway=_Gateway({"wrong": "x"})))
    assert r.status == "halted"
    assert r.halted_at == "scorer"
    assert r.reason.startswith("schema_violation: output:")


async def test_model_unavailable_halts_and_escalates_under_policy():
    @agent(name="scorer", input=Seed, output=Score, model="m1")
    async def scorer(v: Seed) -> Score:
        raise AssertionError

    d = RecordingDispatcher()
    p_on_failure = EscalationPolicy("slack", "q")
    p = Pipeline(name="t", on_failure=p_on_failure, dispatcher=d)
    p.add(scorer)
    r = await p.run(Seed(amount=1), engine=ADKEngine(gateway=_FailingGateway()))
    assert r.status == "escalated"
    assert r.halted_at == "scorer"
    assert len(d.sent) == 1

    # The model-unavailable failure must actually propagate into the escalation,
    # not merely trigger *some* dispatch. RecordingDispatcher.sent holds
    # (package, policy) tuples; assert the configured channel was used and the
    # dispatched package carries the real failure reason.
    package, policy = d.sent[0]
    assert policy is p_on_failure  # the configured policy/channel, not a default
    assert policy.channel == "slack"
    # ModelUnavailableError self-declares halt_reason="model_unavailable", so a
    # regulator reading the package sees the LLM fallback chain was exhausted (not
    # the generic "agent_error"); the model's own message ("down") is appended.
    assert "model_unavailable" in package.reason
    assert "down" in package.reason
    # The run surfaces the same package on RunResult.escalations.
    assert r.escalations[0] is package
    assert "model_unavailable" in r.escalations[0].reason


async def test_deterministic_and_model_agents_compose_in_one_pipeline():
    @agent(name="prep", input=Seed, output=Seed)
    async def prep(v: Seed) -> Seed:
        return Seed(amount=v.amount + 1)

    @agent(name="scorer", input=Seed, output=Score, model="m1")
    async def scorer(v: Seed) -> Score:
        raise AssertionError

    p = Pipeline(name="t")
    p.add(prep)
    p.add(scorer, inputs={"amount": From("prep.amount")})
    r = await p.run(Seed(amount=10), engine=ADKEngine(gateway=_Gateway({"risk": "low"})))
    assert r.status == "completed"
    assert r.outputs["scorer"].risk == "low"


async def test_model_backed_agent_input_is_validated_before_the_model_runs():
    """A model-backed agent's INPUT is validated at the pipeline edge BEFORE the
    engine (and thus the gateway) is reached. When the payload bound into the
    model agent violates its input model, the pipeline halts at the input
    boundary and ``gateway.complete`` is never called.

    Mechanism mirrors ``test_input_schema_violation_halts``: a linear-default
    binding where the predecessor's output is structurally missing a required
    field of the model agent's input. The no-inputs step skips the static check,
    so the violation is caught at runtime input validation — before run_step.
    """

    class Empty(BaseModel):
        # Structurally lacks Seed's required `amount` field, so the linear-default
        # payload handed to `scorer` fails its input validation.
        note: str

    @agent(name="prep", input=Seed, output=Empty)
    async def prep(v: Seed) -> Empty:
        return Empty(note="ok")

    @agent(name="scorer", input=Seed, output=Score, model="m1")
    async def scorer(v: Seed) -> Score:
        raise AssertionError("fn must not run for a model-backed agent")

    class _SpyGateway(LLMGateway):
        def __init__(self):
            self.called = False

        async def complete(self, request):
            self.called = True
            return ModelResponse(output={"risk": "low"}, model_used="m1", raw_text="")

    gw = _SpyGateway()
    p = Pipeline(name="t")
    p.add(prep)
    p.add(scorer)  # linear default; static check skipped for no-inputs steps
    r = await p.run(Seed(amount=1), engine=ADKEngine(gateway=gw))

    assert r.status == "halted"
    assert r.halted_at == "scorer"
    assert r.reason.startswith("schema_violation: input:")
    assert r.steps_run == 1  # prep ran; scorer halted at its input boundary
    # The model was never reached: input validation precedes the gateway call.
    assert gw.called is False
