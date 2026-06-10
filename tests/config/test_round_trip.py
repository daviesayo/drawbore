from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.config import AgentCatalog, from_config, to_config, to_json
from drawbore.evidence import EvidencePolicy
from drawbore.escalation import EscalationPolicy
from drawbore.pipeline import Pipeline
from drawbore.pipeline.binding import From


class TxIn(BaseModel):
    amount: int


class TxOut(BaseModel):
    amount: int
    risk: str = "low"


def _agent(name, *, inp=TxIn, out=TxOut, **spec):
    @agent(name=name, input=inp, output=out, **spec)
    async def fn(v):  # type: ignore[no-untyped-def]
        return out(amount=getattr(v, "amount", 0))
    return fn


def _round_trip(pipeline, cat, registry=None):
    config = to_config(pipeline, agents=cat)
    rebuilt = from_config(config, agents=cat, registry=registry)
    assert to_config(rebuilt, agents=cat) == config
    return config, rebuilt


def test_linear_pipeline_round_trips():                       # acceptance 1
    a, b = _agent("fetch"), _agent("screen")
    cat = AgentCatalog()
    cat.register("pkg:fetch", a)
    cat.register("pkg:screen", b)
    p = Pipeline(name="aml", version="0.1.0")
    p.add(a)
    p.add(b, depends_on=["fetch"])
    _round_trip(p, cat)


def test_fan_in_pipeline_with_bindings_round_trips():         # acceptance 2
    class AOut(BaseModel):
        transaction: int

    class BOut(BaseModel):
        risk_score: int

    class MergeIn(BaseModel):
        transaction: int
        risk: int

    fa = _agent("fetch_transaction", out=AOut)
    fb = _agent("risk_score", out=BOut)

    @agent(name="merge", input=MergeIn, output=MergeIn)
    async def merge(v: MergeIn) -> MergeIn:
        return v

    cat = AgentCatalog()
    cat.register("pkg:fetch", fa)
    cat.register("pkg:risk", fb)
    cat.register("pkg:merge", merge)
    p = Pipeline(name="aml")
    p.add(fa)
    p.add(fb, depends_on=["fetch_transaction"])
    p.add(
        merge,
        inputs={"transaction": From("fetch_transaction.transaction"), "risk": From("risk_score.risk_score")},
        depends_on=["fetch_transaction", "risk_score"],
    )
    config, rebuilt = _round_trip(p, cat)
    binding_step = config.steps[2]
    assert binding_step.input.bindings is not None
    assert binding_step.input.bindings["transaction"].from_ == "fetch_transaction.transaction"
    assert binding_step.input.bindings["risk"].from_ == "risk_score.risk_score"
    assert binding_step.depends_on == ["fetch_transaction", "risk_score"]
    # Verify the REBUILT pipeline's LIVE inputs (not just the re-exported config):
    # round-trip equality detects loss, but the live From refs prove the resolver
    # rewired both bindings to the correct sources, not just one.
    assert rebuilt.steps[2].inputs["transaction"].ref == "fetch_transaction.transaction"
    assert rebuilt.steps[2].inputs["risk"].ref == "risk_score.risk_score"


def test_escalation_and_confidence_round_trip():              # acceptance 25
    a = _agent("fetch")
    cat = AgentCatalog()
    cat.register("pkg:fetch", a)
    p = Pipeline(
        name="aml", confidence_threshold=0.8,
        on_failure=EscalationPolicy(channel="human_review", target="compliance_ops", mode="async"),
    )
    p.add(a)
    config, rebuilt = _round_trip(p, cat)
    assert config.pipeline.confidence_threshold == 0.8
    assert config.pipeline.on_failure.channel == "human_review"
    assert config.pipeline.on_failure.mode == "async"
    assert rebuilt._confidence_threshold == 0.8
    assert rebuilt._on_failure == EscalationPolicy("human_review", "compliance_ops", "async")


def test_per_step_evidence_policy_round_trips_and_is_applied():   # acceptance 26
    a = _agent("screen", model="gpt-4o")                      # model-backed for evidence
    cat = AgentCatalog()
    cat.register("pkg:screen", a)
    policy = EvidencePolicy(
        name="aml_rows", enabled=True, mode="compress",
        allowed_transforms=("json_rows",), min_tokens=800, max_output_tokens=400,
        require_original_store=True, allow_full_retrieval=False,
        allow_search_retrieval=True, ttl_seconds=None, strict=False,
    )
    p = Pipeline(name="aml")
    p.add(a, evidence=policy)
    config, rebuilt = _round_trip(p, cat)
    assert config.steps[0].evidence.name == "aml_rows"
    assert config.steps[0].evidence.allowed_transforms == ("json_rows",)
    assert rebuilt.steps[0].evidence == policy                # applied to Step.evidence


def test_one_shot_model_metadata_round_trips():              # acceptance 27
    a = _agent("writer", model="gpt-4o", fallback_model="gpt-4o-mini", instructions="score it")
    cat = AgentCatalog()
    cat.register("pkg:writer", a)
    p = Pipeline(name="aml")
    p.add(a)
    config, _ = _round_trip(p, cat)
    assert config.agents[0].model == "gpt-4o"
    assert config.agents[0].fallback_model == "gpt-4o-mini"
    assert config.agents[0].instructions == "score it"


def test_model_plus_tools_without_fallback_round_trips():    # acceptance 28
    from drawbore.tools import ToolRegistry
    reg = ToolRegistry()
    reg.register_tool("risk.lookup", lambda args: {}, allowed_operations=("invoke",))
    a = _agent("solver", model="gpt-4o", tools=["risk.lookup"])   # no fallback
    cat = AgentCatalog()
    cat.register("pkg:solver", a)
    p = Pipeline(name="aml", registry=reg)
    p.add(a)
    config, _ = _round_trip(p, cat, registry=reg)
    assert config.agents[0].model == "gpt-4o"
    assert config.agents[0].tools == ["risk.lookup"]
    assert config.agents[0].fallback_model is None


def test_to_json_is_deterministic_across_calls():           # acceptance 29
    a, b = _agent("fetch"), _agent("screen")
    cat = AgentCatalog()
    cat.register("pkg:fetch", a)
    cat.register("pkg:screen", b)
    p = Pipeline(name="aml")
    p.add(a)
    p.add(b, depends_on=["fetch"])
    # Same object twice — guards against an accidental sort_keys=False regression.
    assert to_json(p, agents=cat) == to_json(p, agents=cat)
    # Stronger: a DISTINCT live pipeline rebuilt from the manifest serializes to the
    # identical canonical bytes — proves the serializer's object-graph traversal is
    # order-independent, not just stable for one in-memory instance.
    rebuilt = from_config(to_config(p, agents=cat), agents=cat)
    assert to_json(rebuilt, agents=cat) == to_json(p, agents=cat)
