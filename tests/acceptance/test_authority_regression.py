"""Acceptance tests: authority-regression proof on a serialized fixture.

Task 7: build a 2-step pipeline manifest with ``to_config`` in two variants
(allow_full=False / True); assert the diff, the CI-gate raise, and narrowing.

Task 8: run the same fixture through ``test_mode`` and verify the runtime
exercised no ref outside the static footprint (soundness check).

The fixture is intentionally NOT coupled to the locked remittance acceptance
suite; it proves the same property (flip ``allow_full_retrieval`` → ``.added``
is exactly the full evidence fact) with a focused, stable fixture.
"""

import pytest
from pydantic import BaseModel

from drawbore import Pipeline, agent, From
from drawbore.config import AgentCatalog, to_config, authority_diff, check_no_new_authority
from drawbore.config.authority import CapabilityFact
from drawbore.config.errors import AuthorityRegressionError
from drawbore.evidence import EvidencePolicy
from drawbore.testing import assert_footprint_sound


# --- minimal Pydantic models ---

class Txn(BaseModel):
    id: str


class Risk(BaseModel):
    risk_level: str


# --- the two fixture agents ---

@agent(name="loader", input=Txn, output=Txn)
async def loader(t: Txn) -> Txn:
    return t


@agent(name="risk_scorer", input=Txn, output=Risk, model="risk-1.0")
async def risk_scorer(t: Txn) -> Risk: ...  # one-shot model agent; body unused


# --- fixture builder ---

def _build(*, allow_full: bool):
    """Build the pipeline + catalog and serialize to a PipelineConfig manifest."""
    policy = EvidencePolicy(
        name="ev", enabled=True, mode="compress",
        allowed_transforms=("json_rows",), min_tokens=1,
        allow_search_retrieval=True, allow_full_retrieval=allow_full,
    )
    pipeline = (
        Pipeline("authzfix")
        .add(loader)
        .add(risk_scorer, inputs={"id": From("loader.id")}, evidence=policy)
    )
    catalog = AgentCatalog()
    catalog.register("x.loader", loader)
    catalog.register("x.risk_scorer", risk_scorer)
    return to_config(pipeline, agents=catalog)


# --- Task 7: manifest diff + CI gate ---

def test_widening_evidence_to_full_is_a_regression():
    old = _build(allow_full=False)
    new = _build(allow_full=True)
    d = authority_diff(old, new)
    assert d.ok is False
    assert d.added == frozenset({
        CapabilityFact("risk_scorer", "evidence", "evidence://retrieve", "full")
    })
    with pytest.raises(AuthorityRegressionError):
        check_no_new_authority(old, new)


def test_identity_and_narrowing_pass():
    full = _build(allow_full=True)
    assert authority_diff(full, full).ok is True
    # narrowing (full -> search only) removes the 'full' fact, adds nothing
    narrowed = authority_diff(full, _build(allow_full=False))
    assert narrowed.ok is True
    assert CapabilityFact("risk_scorer", "evidence", "evidence://retrieve", "full") in narrowed.removed


# --- Task 8: soundness pass over a real test_mode run ---

async def test_real_run_stays_within_its_footprint():
    config = _build(allow_full=False)
    pipeline = (
        Pipeline("authzfix")
        .add(loader)
        .add(
            risk_scorer,
            inputs={"id": From("loader.id")},
            evidence=EvidencePolicy(
                name="ev", enabled=True, mode="compress",
                allowed_transforms=("json_rows",), min_tokens=1,
                allow_search_retrieval=True, allow_full_retrieval=False,
            ),
        )
    )
    async with pipeline.test_mode(
        mock_model_responses={"risk_scorer": {"risk_level": "low"}},
    ) as tp:
        result = await tp.run(Txn(id="t-1"))
    assert result.status == "completed"
    # The run exercised no tool/evidence ref outside the manifest's static footprint.
    assert_footprint_sound(config, result.audit_trace)
