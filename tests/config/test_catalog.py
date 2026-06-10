import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.config import AgentCatalog, ConfigResolutionError
from drawbore.config.catalog import ref_for, resolve_ref


class In(BaseModel):
    x: int


class Out(BaseModel):
    y: int


def _agent(name):
    @agent(name=name, input=In, output=Out)
    async def fn(v: In) -> Out:
        return Out(y=v.x)
    return fn


def test_catalog_resolves_a_registered_ref():
    a = _agent("screen")
    cat = AgentCatalog()
    cat.register("pkg:screen", a)
    assert cat.resolve("pkg:screen") is a


def test_catalog_ref_for_returns_the_registered_ref():
    a = _agent("screen")
    cat = AgentCatalog()
    cat.register("pkg:screen", a)
    assert cat.ref_for(a) == "pkg:screen"


def test_catalog_resolve_missing_ref_fails_closed():
    cat = AgentCatalog()
    with pytest.raises(ConfigResolutionError, match="agent ref 'pkg:missing' is not registered"):
        cat.resolve("pkg:missing")


def test_catalog_ref_for_uncataloged_agent_fails_closed():
    a = _agent("screen")
    cat = AgentCatalog()
    with pytest.raises(ConfigResolutionError, match="not registered in the catalog"):
        cat.ref_for(a)


def test_register_rejects_same_agent_under_a_second_ref():
    # Exactly one ref per agent — a silent last-write would corrupt export.
    a = _agent("screen")
    cat = AgentCatalog()
    cat.register("pkg:screen", a)
    with pytest.raises(ConfigResolutionError, match="already registered under ref 'pkg:screen'"):
        cat.register("pkg:other", a)


def test_register_same_ref_and_agent_is_idempotent():
    a = _agent("screen")
    cat = AgentCatalog()
    cat.register("pkg:screen", a)
    cat.register("pkg:screen", a)          # identical pair → no-op, no raise
    assert cat.resolve("pkg:screen") is a
    assert cat.ref_for(a) == "pkg:screen"


def test_register_rejects_ref_reused_for_a_different_agent():
    a, b = _agent("screen"), _agent("other")
    cat = AgentCatalog()
    cat.register("pkg:x", a)
    with pytest.raises(ConfigResolutionError, match="already registered to a different agent"):
        cat.register("pkg:x", b)


def test_resolve_ref_accepts_catalog_or_mapping():
    a = _agent("screen")
    cat = AgentCatalog()
    cat.register("pkg:screen", a)
    assert resolve_ref(cat, "pkg:screen") is a
    assert resolve_ref({"pkg:screen": a}, "pkg:screen") is a


def test_resolve_ref_missing_in_mapping_fails_closed():
    with pytest.raises(ConfigResolutionError, match="agent ref 'pkg:missing' is not registered"):
        resolve_ref({}, "pkg:missing")


def test_ref_for_mapping_zero_matches_fails_closed():
    a = _agent("screen")
    with pytest.raises(ConfigResolutionError, match="not in the agent mapping"):
        ref_for({}, a)


def test_ref_for_mapping_multiple_matches_fails_closed():
    a = _agent("screen")
    with pytest.raises(ConfigResolutionError, match="no unique catalog ref"):
        ref_for({"r1": a, "r2": a}, a)


def test_ref_for_mapping_unique_match():
    a = _agent("screen")
    assert ref_for({"pkg:screen": a}, a) == "pkg:screen"
