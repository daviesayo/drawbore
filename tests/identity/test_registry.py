import dataclasses
import pytest
from pydantic import BaseModel
from drawbore.agent import agent
from drawbore.identity import AgentIdentity, IdentityError, IdentityRegistry


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


def _spec(name="scorer", risk="low", tools=()):
    @agent(name=name, input=In, output=Out, risk_tier=risk, tools=list(tools))
    async def fn(v: In) -> Out:
        return Out(y=v.x)
    return fn.spec


def test_register_requires_a_sponsor():
    reg = IdentityRegistry()
    with pytest.raises(IdentityError):
        reg.register(_spec(), sponsor="", purpose="p", ttl_seconds=60)


def test_register_issues_active_identity_and_is_runnable():
    reg = IdentityRegistry()
    idn = reg.register(
        _spec(), sponsor="alice@bank", purpose="score", ttl_seconds=3600,
        pipeline="payments", tenant="acme", agent_id="id-1",
        now=lambda: "2026-06-03T00:00:00Z",
    )
    assert isinstance(idn, AgentIdentity)
    assert idn.agent_id == "id-1" and idn.sponsor == "alice@bank"
    assert reg.is_registered("scorer")
    assert reg.state_of("scorer") == "active"
    assert reg.run_block_reason("scorer") is None  # active + attested → runnable


def test_double_register_is_rejected():
    reg = IdentityRegistry()
    reg.register(_spec(), sponsor="a", purpose="p", ttl_seconds=60)
    with pytest.raises(IdentityError):
        reg.register(_spec(), sponsor="a", purpose="p", ttl_seconds=60)


def test_unregistered_agent_is_draft_and_not_gated():
    reg = IdentityRegistry()
    assert reg.is_registered("ghost") is False
    assert reg.state_of("ghost") == "draft"
    assert reg.run_block_reason("ghost") is None  # draft (unregistered): schema-only path


def test_suspend_blocks_then_resume_unblocks():
    reg = IdentityRegistry()
    reg.register(_spec(), sponsor="a", purpose="p", ttl_seconds=60)
    reg.suspend("scorer")
    assert reg.state_of("scorer") == "suspended"
    assert reg.run_block_reason("scorer") == "suspended"
    reg.resume("scorer")
    assert reg.state_of("scorer") == "active"
    assert reg.run_block_reason("scorer") is None


def test_decommission_is_terminal_and_blocks():
    reg = IdentityRegistry()
    reg.register(_spec(), sponsor="a", purpose="p", ttl_seconds=60)
    reg.decommission("scorer")
    assert reg.state_of("scorer") == "decommissioned"
    assert reg.run_block_reason("scorer") == "decommissioned"
    with pytest.raises(IdentityError):
        reg.resume("scorer")  # cannot leave decommissioned
    with pytest.raises(IdentityError):
        reg.suspend("scorer")


def test_resume_requires_suspended_state():
    reg = IdentityRegistry()
    reg.register(_spec(), sponsor="a", purpose="p", ttl_seconds=60)
    with pytest.raises(IdentityError):
        reg.resume("scorer")  # already active


def test_surface_change_requires_reattestation_and_blocks():
    reg = IdentityRegistry()
    reg.register(_spec(risk="low"), sponsor="alice", purpose="p", ttl_seconds=60)
    reg.update(_spec(risk="critical"))  # risk tier escalated → surface changed
    assert reg.run_block_reason("scorer") == "reattestation_required"
    with pytest.raises(IdentityError):
        reg.reattest("scorer", sponsor="mallory", spec=_spec(risk="critical"))
    reg.reattest("scorer", sponsor="alice", spec=_spec(risk="critical"))
    assert reg.run_block_reason("scorer") is None


def test_reattest_with_wrong_surface_is_rejected():
    # The block was triggered by escalating risk to "critical"; re-attesting against
    # the original "low" surface must NOT clear it (no privilege laundering).
    reg = IdentityRegistry()
    reg.register(_spec(risk="low"), sponsor="alice", purpose="p", ttl_seconds=60)
    reg.update(_spec(risk="critical"))
    with pytest.raises(IdentityError):
        reg.reattest("scorer", sponsor="alice", spec=_spec(risk="low"))
    assert reg.run_block_reason("scorer") == "reattestation_required"  # still blocked
    # the correct surface clears it
    reg.reattest("scorer", sponsor="alice", spec=_spec(risk="critical"))
    assert reg.run_block_reason("scorer") is None


def test_reattest_after_decommission_is_rejected():
    reg = IdentityRegistry()
    reg.register(_spec(), sponsor="alice", purpose="p", ttl_seconds=60)
    reg.decommission("scorer")
    with pytest.raises(IdentityError):
        reg.reattest("scorer", sponsor="alice", spec=_spec())


def test_reattestation_block_survives_suspend_resume():
    reg = IdentityRegistry()
    reg.register(_spec(risk="low"), sponsor="alice", purpose="p", ttl_seconds=60)
    reg.update(_spec(risk="critical"))   # block
    reg.suspend("scorer")
    reg.resume("scorer")
    assert reg.run_block_reason("scorer") == "reattestation_required"  # preserved


def test_register_rejects_a_whitespace_only_sponsor():
    reg = IdentityRegistry()
    with pytest.raises(IdentityError):
        reg.register(_spec(), sponsor="   ", purpose="p", ttl_seconds=60)


def test_update_with_identical_surface_does_not_trigger_reattestation():
    reg = IdentityRegistry()
    reg.register(_spec(), sponsor="a", purpose="p", ttl_seconds=60)
    reg.update(_spec())  # same surface
    assert reg.run_block_reason("scorer") is None


def test_lifecycle_calls_on_unknown_agent_raise():
    reg = IdentityRegistry()
    for op in ("suspend", "resume", "decommission"):
        with pytest.raises(IdentityError):
            getattr(reg, op)("nobody")
