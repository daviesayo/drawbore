from pydantic import BaseModel, create_model
from drawbore.agent import agent
from drawbore.identity import AgentIdentity, attestation_surface


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


def test_agent_identity_carries_brief_attributes():
    idn = AgentIdentity(
        agent_id="agent-1",
        name="scorer",
        purpose="score risk",
        risk_tier="high",
        sponsor="alice@bank.example",
        ttl_seconds=3600,
        pipeline="payments",
        tenant="acme",
        created_at="2026-06-03T00:00:00Z",
    )
    assert idn.agent_id == "agent-1"
    assert idn.sponsor == "alice@bank.example"
    assert idn.risk_tier == "high"
    assert idn.ttl_seconds == 3600
    assert idn.pipeline == "payments" and idn.tenant == "acme"
    # frozen value object
    import dataclasses, pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        idn.sponsor = "mallory@evil.example"


def test_attestation_surface_changes_with_risk_tier_tools_and_schema():
    @agent(name="a", input=In, output=Out, risk_tier="low", tools=["db"])
    async def a(v: In) -> Out:
        return Out(y=v.x)

    base = attestation_surface(a.spec)
    assert base == attestation_surface(a.spec)  # deterministic

    @agent(name="a", input=In, output=Out, risk_tier="high", tools=["db"])
    async def a2(v: In) -> Out:
        return Out(y=v.x)

    assert attestation_surface(a2.spec) != base  # risk tier change moves the surface

    class Out2(BaseModel):
        y: int
        note: str

    @agent(name="a", input=In, output=Out2, risk_tier="low", tools=["db"])
    async def a3(v: In) -> Out2:
        return Out2(y=v.x, note="n")

    assert attestation_surface(a3.spec) != base  # output schema change moves the surface


def test_attestation_surface_is_tool_order_independent():
    @agent(name="t1", input=In, output=Out, tools=["a", "b"])
    async def t1(v: In) -> Out:
        return Out(y=v.x)

    @agent(name="t2", input=In, output=Out, tools=["b", "a"])
    async def t2(v: In) -> Out:
        return Out(y=v.x)

    assert attestation_surface(t1.spec) == attestation_surface(t2.spec)


def _spec_with_nested(extra_nested_field: bool):
    # The nested model has the SAME class name in both versions; only its
    # structure differs — the realistic "redefined Address in place" scenario.
    nested_fields = {"street": (str, ...)}
    if extra_nested_field:
        nested_fields["access_code"] = (str, ...)
    Address = create_model("Address", **nested_fields)
    CustomerIn = create_model("CustomerIn", address=(Address, ...))

    @agent(name="cust", input=CustomerIn, output=Out)
    async def fn(v: CustomerIn) -> Out:
        return Out(y=1)

    return fn.spec


def test_attestation_surface_detects_nested_model_change():
    # A structural change to a NESTED input model must move the surface,
    # otherwise a schema change silently escapes re-attestation.
    s1 = _spec_with_nested(False)
    s2 = _spec_with_nested(True)
    assert attestation_surface(s1) != attestation_surface(s2)


def test_attestation_surface_changes_with_input_schema():
    @agent(name="i1", input=In, output=Out)
    async def i1(v: In) -> Out:
        return Out(y=v.x)

    class In2(BaseModel):
        x: int
        extra: str

    @agent(name="i1", input=In2, output=Out)
    async def i2(v: In2) -> Out:
        return Out(y=v.x)

    assert attestation_surface(i1.spec) != attestation_surface(i2.spec)


def test_attestation_surface_changes_when_a_tool_is_added_or_removed():
    @agent(name="t", input=In, output=Out, tools=["a"])
    async def t_one(v: In) -> Out:
        return Out(y=v.x)

    @agent(name="t", input=In, output=Out, tools=["a", "b"])
    async def t_two(v: In) -> Out:
        return Out(y=v.x)

    assert attestation_surface(t_one.spec) != attestation_surface(t_two.spec)
