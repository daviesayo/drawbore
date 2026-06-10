# tests/config/test_graph_round_trip.py
from drawbore.config.models import PipelineConfig


def test_legacy_manifest_without_kind_loads():
    # a legacy manifest (no `kind` on steps) must still validate (mode=before injection)
    raw = {
        "schema_version": 1,
        "pipeline": {"name": "p", "version": "1.0.0"},
        "agents": [{
            "name": "a", "ref": "x.a", "version": "0.0.0", "risk_tier": "low",
            "requires_human_approval": False, "context_access": "none", "tools": [],
            "model": None, "fallback_model": None, "instructions": None,
            "input_schema": {}, "output_schema": {},
            "input_schema_hash": "h", "output_schema_hash": "h",
        }],
        "steps": [{"agent": "a", "input": {"source": "initial"}, "depends_on": []}],
    }
    cfg = PipelineConfig.model_validate(raw)
    assert cfg.steps[0].kind == "agent"


def test_default_node_kind_validator_is_pure():
    # D68a purity: the before-validator must NOT mutate the caller's dict.
    # Build a kind-less step dict and keep a direct reference to it.
    step_dict = {"agent": "a", "input": {"source": "initial"}, "depends_on": []}
    raw = {
        "schema_version": 1,
        "pipeline": {"name": "p", "version": "1.0.0"},
        "agents": [{
            "name": "a", "ref": "x.a", "version": "0.0.0", "risk_tier": "low",
            "requires_human_approval": False, "context_access": "none", "tools": [],
            "model": None, "fallback_model": None, "instructions": None,
            "input_schema": {}, "output_schema": {},
            "input_schema_hash": "h", "output_schema_hash": "h",
        }],
        "steps": [step_dict],
    }
    cfg = PipelineConfig.model_validate(raw)
    # Validator must have injected kind into the *parsed* model...
    assert cfg.steps[0].kind == "agent"
    # ...but must NOT have mutated the original step dict.
    assert "kind" not in step_dict, (
        "_default_node_kind mutated the caller's dict in place (purity violation)"
    )


def test_join_node_parses_by_kind():
    raw = {
        "schema_version": 1,
        "pipeline": {"name": "p", "version": "1.0.0"},
        "agents": [],
        "steps": [{
            "kind": "join", "name": "review", "sources": ["a", "b"],
            "policy": "exactly_one", "output_schema": {}, "output_schema_hash": "h",
        }],
    }
    cfg = PipelineConfig.model_validate(raw)
    assert cfg.steps[0].kind == "join" and cfg.steps[0].policy == "exactly_one"


import pytest
from drawbore.config import AgentCatalog, from_json, to_json
from drawbore.config.errors import ConfigResolutionError
from tests.pipeline.test_join import (  # reuse the agents/pipeline
    score, enhanced, standard, writer, Seed, Review, Merged,
)
from drawbore.pipeline import Pipeline, From, When, Join


def _catalog():
    c = AgentCatalog()
    c.register("x.score", score); c.register("x.enhanced", enhanced)
    c.register("x.standard", standard); c.register("x.writer", writer)
    return c


def _pipe():
    p = Pipeline(name="join")
    p.add(score)
    p.add(enhanced, inputs={"level": From("score.level")}, when=When("score.level", equals="high"))
    p.add(standard, inputs={"level": From("score.level")}, when=When("score.level", in_=("low", "medium")))
    p.add(Join("review", sources=["enhanced", "standard"], policy="exactly_one", output=Review))
    p.add(writer, inputs={"verdict": From("review.verdict")})
    return p


async def test_branch_join_round_trip_runs():
    cat = _catalog()
    manifest = to_json(_pipe(), agents=cat)
    assert '"kind":"join"' in manifest
    rebuilt = from_json(manifest, agents=cat)
    async with rebuilt.test_mode() as test:
        r = await test.run(Seed(value=1))
    assert r.status == "completed" and r.outputs["writer"].verdict == "standard"


def _all_present_pipe():
    """A pipeline whose fan-in is an all_present join that merges two upstream
    agents into a DISTINCT output model (Merged) via `inputs` bindings."""
    p = Pipeline(name="all-present-rt")
    p.add(score)
    p.add(enhanced, inputs={"level": From("score.level")})
    p.add(standard, inputs={"level": From("score.level")})
    p.add(Join(
        "review",
        sources=["enhanced", "standard"],
        policy="all_present",
        output=Merged,
        inputs={
            "enhanced_verdict": From("enhanced.verdict"),
            "standard_verdict": From("standard.verdict"),
        },
    ))
    return p


def _all_present_catalog():
    c = AgentCatalog()
    c.register("x.score", score)
    c.register("x.enhanced", enhanced)
    c.register("x.standard", standard)
    return c


def test_all_present_join_export_succeeds_import_fails_closed():
    """to_json must succeed (the manifest captures the shape); from_json must fail
    CLOSED with ConfigResolutionError because the all_present output model is built
    from bindings and cannot be recovered from the manifest."""
    cat = _all_present_catalog()
    # Export must still work — the manifest records the shape.
    manifest = to_json(_all_present_pipe(), agents=cat)
    assert '"all_present"' in manifest

    # Import must fail closed, not silently mis-recover.
    with pytest.raises(ConfigResolutionError, match="all_present"):
        from_json(manifest, agents=cat)
