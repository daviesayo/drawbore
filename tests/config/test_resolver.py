import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.config import (
    AgentCatalog,
    ConfigResolutionError,
    PipelineConfig,
    from_config,
    from_json,
    to_config,
)
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


def _linear_baseline():
    a, b = _agent("fetch"), _agent("screen")
    cat = AgentCatalog()
    cat.register("pkg:fetch", a)
    cat.register("pkg:screen", b)
    p = Pipeline(name="aml", version="0.1.0")
    p.add(a)
    p.add(b, depends_on=["fetch"])
    return to_config(p, agents=cat), cat


def _mutate(config: PipelineConfig, fn) -> dict:
    data = config.model_dump(mode="json", by_alias=True)
    fn(data)
    return data


def test_from_config_rebuilds_a_linear_pipeline():
    config, cat = _linear_baseline()
    rebuilt = from_config(config, agents=cat)
    assert isinstance(rebuilt, Pipeline)
    assert [s.agent.name for s in rebuilt.steps] == ["fetch", "screen"]
    assert rebuilt.steps[1].depends_on == ["fetch"]


def test_rejects_unknown_schema_version():
    config, cat = _linear_baseline()
    data = _mutate(config, lambda d: d.__setitem__("schema_version", 2))
    with pytest.raises(ConfigResolutionError, match="unknown config schema_version 2"):
        from_json(data, agents=cat)


def test_rejects_missing_agent_ref():
    config, cat = _linear_baseline()
    data = _mutate(config, lambda d: d["agents"][0].__setitem__("ref", "pkg:missing"))
    with pytest.raises(ConfigResolutionError, match="agent ref 'pkg:missing' is not registered"):
        from_json(data, agents=cat)


def test_rejects_unknown_step_agent():
    config, cat = _linear_baseline()
    data = _mutate(config, lambda d: d["steps"][1].__setitem__("agent", "approve"))
    with pytest.raises(ConfigResolutionError, match="references unknown agent 'approve'"):
        from_json(data, agents=cat)


def test_rejects_missing_registered_tool():
    from drawbore.tools import ToolRegistry
    # A local registry (no global-state pollution); register the tool BEFORE add,
    # because Pipeline.add itself checks tool registration.
    reg = ToolRegistry()
    reg.register_tool("transactions.read", lambda args: {}, allowed_operations=("invoke",))
    a = _agent("screen", tools=["transactions.read"])
    cat = AgentCatalog()
    cat.register("pkg:screen", a)
    p = Pipeline(name="aml", registry=reg)
    p.add(a)
    config = to_config(p, agents=cat)
    # resolve against a registry WITHOUT the tool → fail closed
    empty = ToolRegistry()
    with pytest.raises(ConfigResolutionError, match="declared tool 'transactions.read' is not registered"):
        from_config(config, agents=cat, registry=empty)


def test_rejects_agent_version_drift():
    config, cat = _linear_baseline()
    data = _mutate(config, lambda d: d["agents"][0].__setitem__("version", "9.9.9"))
    with pytest.raises(ConfigResolutionError, match="version drift"):
        from_json(data, agents=cat)


def test_rejects_risk_tier_drift():
    config, cat = _linear_baseline()
    data = _mutate(config, lambda d: d["agents"][0].__setitem__("risk_tier", "critical"))
    with pytest.raises(ConfigResolutionError, match="risk.?tier drift"):
        from_json(data, agents=cat)


def test_rejects_tool_declaration_drift():
    config, cat = _linear_baseline()
    data = _mutate(config, lambda d: d["agents"][0].__setitem__("tools", ["transactions.read"]))
    with pytest.raises(ConfigResolutionError, match="tool drift"):
        from_json(data, agents=cat)


def test_rejects_input_schema_drift():
    config, cat = _linear_baseline()
    data = _mutate(config, lambda d: d["agents"][0].__setitem__("input_schema_hash", "sha256:deadbeef"))
    with pytest.raises(ConfigResolutionError, match="input schema drift"):
        from_json(data, agents=cat)


def test_rejects_output_schema_drift():
    config, cat = _linear_baseline()
    data = _mutate(config, lambda d: d["agents"][0].__setitem__("output_schema_hash", "sha256:deadbeef"))
    with pytest.raises(ConfigResolutionError, match="output schema drift"):
        from_json(data, agents=cat)


def test_rejects_duplicate_agent_names_in_manifest():
    config, cat = _linear_baseline()

    def dup(d):
        d["agents"][1]["name"] = "fetch"      # two agents both named 'fetch'
    data = _mutate(config, dup)
    with pytest.raises(ConfigResolutionError, match="duplicate agent name 'fetch'"):
        from_json(data, agents=cat)


def test_model_tools_fallback_combo_now_loads():
    # model + tools + fallback_model is no longer rejected at import.
    # Previously this asserted a ConfigResolutionError ("tools and
    # fallback_model"); now the manifest round-trips and rebuilds.
    from drawbore.tools import ToolRegistry
    reg = ToolRegistry()
    reg.register_tool("transactions.read", lambda args: {}, allowed_operations=("invoke",))
    a = _agent("solver", model="gpt-4o", fallback_model="gpt-4o-mini", tools=["transactions.read"])
    cat = AgentCatalog()
    cat.register("pkg:solver", a)
    p = Pipeline(name="aml", registry=reg)
    p.add(a)
    config = to_config(p, agents=cat)
    rebuilt = from_config(config, agents=cat, registry=reg)
    assert rebuilt is not None


def test_source_initial_only_valid_for_first_step_with_empty_depends_on():
    config, cat = _linear_baseline()
    # make step 1 (non-first) claim source='initial'
    data = _mutate(config, lambda d: d["steps"][1].__setitem__("input", {"source": "initial"}))
    with pytest.raises(ConfigResolutionError, match="source='initial'"):
        from_json(data, agents=cat)


def test_source_previous_must_name_immediate_predecessor():
    config, cat = _linear_baseline()
    data = _mutate(config, lambda d: d["steps"][1]["input"].__setitem__("agent", "nobody"))
    with pytest.raises(ConfigResolutionError, match="immediate predecessor 'fetch'"):
        from_json(data, agents=cat)


def test_bindings_require_matching_depends_on():
    class FanOut(BaseModel):
        amount: int

    a = _agent("fetch", out=FanOut)

    @agent(name="merge", input=TxIn, output=TxIn)
    async def merge(v: TxIn) -> TxIn:
        return v

    cat = AgentCatalog()
    cat.register("pkg:fetch", a)
    cat.register("pkg:merge", merge)
    p = Pipeline(name="aml")
    p.add(a)
    p.add(merge, inputs={"amount": From("fetch.amount")}, depends_on=["fetch"])
    config = to_config(p, agents=cat)
    # break depends_on so it no longer matches the binding source
    data = config.model_dump(mode="json", by_alias=True)
    data["steps"][1]["depends_on"] = []
    with pytest.raises(ConfigResolutionError, match="binding sources"):
        from_json(data, agents=cat)


def test_rejects_literal_binding():
    config, cat = _linear_baseline()

    def make_literal(d):
        d["steps"][1]["input"] = {"bindings": {"amount": {"literal": 5}}}
        d["steps"][1]["depends_on"] = []
    data = _mutate(config, make_literal)
    with pytest.raises(ConfigResolutionError, match="literal input bindings are not supported"):
        from_json(data, agents=cat)


def test_static_incompatible_binding_fails_closed():
    # Pipeline.add runs the static type check, so a lying pipeline cannot be
    # BUILT directly. Build a VALID baseline (int -> int), export it, then mutate the
    # manifest binding to point at an incompatible source field (str -> int). The
    # incompatibility is only discovered when from_config calls Pipeline.add, which
    # the resolver surfaces as ConfigResolutionError.
    class FetchOut(BaseModel):
        amount: int
        name: str

    class ScreenIn(BaseModel):
        amount: int

    fetch = _agent("fetch", out=FetchOut)

    @agent(name="screen", input=ScreenIn, output=ScreenIn)
    async def screen(v: ScreenIn) -> ScreenIn:
        return v

    cat = AgentCatalog()
    cat.register("pkg:fetch", fetch)
    cat.register("pkg:screen", screen)
    p = Pipeline(name="aml")
    p.add(fetch)
    p.add(screen, inputs={"amount": From("fetch.amount")}, depends_on=["fetch"])  # valid
    config = to_config(p, agents=cat)

    data = config.model_dump(mode="json", by_alias=True)
    data["steps"][1]["input"]["bindings"]["amount"]["from"] = "fetch.name"  # str -> int
    with pytest.raises(ConfigResolutionError):
        from_json(data, agents=cat)


def test_rejects_requires_human_approval_drift():
    config, cat = _linear_baseline()
    data = _mutate(config, lambda d: d["agents"][0].__setitem__("requires_human_approval", True))
    with pytest.raises(ConfigResolutionError, match="requires_human_approval drift"):
        from_json(data, agents=cat)


def test_rejects_model_drift():
    config, cat = _linear_baseline()
    data = _mutate(config, lambda d: d["agents"][0].__setitem__("model", "gpt-4o"))
    with pytest.raises(ConfigResolutionError, match="model drift"):
        from_json(data, agents=cat)


def test_rejects_fallback_model_drift():
    config, cat = _linear_baseline()
    data = _mutate(config, lambda d: d["agents"][0].__setitem__("fallback_model", "gpt-4o-mini"))
    with pytest.raises(ConfigResolutionError, match="fallback_model drift"):
        from_json(data, agents=cat)


def test_rejects_instructions_drift():
    config, cat = _linear_baseline()
    data = _mutate(config, lambda d: d["agents"][0].__setitem__("instructions", "do the thing"))
    with pytest.raises(ConfigResolutionError, match="instructions drift"):
        from_json(data, agents=cat)


def test_rejects_two_steps_referencing_the_same_agent():
    # _reject_duplicate_agent_names guards the agents[] declarations; a manifest where
    # two STEPS reference the same agent name is still fail-closed, surfaced from
    # Pipeline.add's duplicate-name guard via the resolver's except wrap.
    config, cat = _linear_baseline()
    data = _mutate(config, lambda d: d["steps"][1].__setitem__("agent", "fetch"))
    with pytest.raises(ConfigResolutionError, match="already has a node named"):
        from_json(data, agents=cat)


def test_from_json_rejects_malformed_json_as_config_error():
    # from_json's contract is fail-closed: it raises ONLY ConfigResolutionError,
    # so a syntactically invalid JSON string must not escape as json.JSONDecodeError.
    _, cat = _linear_baseline()
    with pytest.raises(ConfigResolutionError, match="not valid JSON"):
        from_json("{ this is not json", agents=cat)
