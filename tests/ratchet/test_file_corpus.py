# tests/ratchet/test_file_corpus.py
"""Tests for FileRegressionCorpus — file-backed, cross-restart persistence.

Covers: append + verify, reload in order, enum-identity reconstruction,
tamper detection, sponsor round-trip, and end-to-end admit() against the file
corpus (the load-bearing case that proves the Containment enum is correctly
reconstructed on reload).
"""
import json

import pytest

from drawbore.ratchet import (
    CorpusIntegrityError,
    FileRegressionCorpus,
    RatchetError,
    SafetyProperty,
    seal_case,
)
from drawbore.testing import Containment, ContainmentCase

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PROP = SafetyProperty(
    dim="run_status",
    step="risk_scorer",
    assertion="neq",
    value="completed",
    description=(
        "a below-threshold confidence from 'risk_scorer' must not complete the run"
    ),
)
CC = ContainmentCase(
    name="low_confidence:risk_scorer",
    kind="low_confidence",
    target="risk_scorer",
    payload={"risk_level": "low", "confidence": 0.0, "factors": []},
    expect=Containment.HALTED,
)


def _case(case_id: str = "c-001", predecessor: str = "genesis") -> object:
    return seal_case(
        case_id=case_id,
        property=PROP,
        containment_case=CC,
        baseline_fingerprint="sha256:" + "aa" * 32,
        initial_hash="sha256:" + "bb" * 32,
        baseline_mocks_hash="sha256:" + "cc" * 32,
        derived_at="2026-06-13T00:00:00Z",
        predecessor_hash=predecessor,
    )


# ---------------------------------------------------------------------------
# Append + verify
# ---------------------------------------------------------------------------


def test_append_two_cases_and_verify_passes(tmp_path):
    corpus = FileRegressionCorpus(tmp_path)
    c1 = _case()
    corpus.append(c1, sponsor="alice")
    corpus.verify()  # single case — must not raise

    c2 = _case(case_id="c-002", predecessor=c1.case_hash)
    corpus.append(c2, sponsor="bob")
    corpus.verify()  # two cases — chain must hold


def test_cases_and_root_match_append_order(tmp_path):
    corpus = FileRegressionCorpus(tmp_path)
    c1 = _case()
    corpus.append(c1, sponsor="alice")
    c2 = _case(case_id="c-002", predecessor=c1.case_hash)
    corpus.append(c2, sponsor="bob")

    assert [c.case_id for c in corpus.cases()] == ["c-001", "c-002"]
    assert corpus.root() == c2.case_hash


# ---------------------------------------------------------------------------
# Reload: order, chain, enum identity
# ---------------------------------------------------------------------------


def test_reload_preserves_order_and_verify_passes(tmp_path):
    corpus = FileRegressionCorpus(tmp_path)
    c1 = _case()
    corpus.append(c1, sponsor="alice")
    c2 = _case(case_id="c-002", predecessor=c1.case_hash)
    corpus.append(c2, sponsor="bob")

    fresh = FileRegressionCorpus(tmp_path)
    loaded = fresh.cases()
    assert len(loaded) == 2
    assert loaded[0].case_id == "c-001"
    assert loaded[1].case_id == "c-002"
    assert loaded[1].predecessor_hash == c1.case_hash
    fresh.verify()  # must not raise


def test_enum_identity_on_reload(tmp_path):
    """The load-bearing correctness invariant: ContainmentCase.expect must be
    reconstructed as a Containment ENUM MEMBER, not a plain string.

    admit() compares the observed verdict against expect with IS NOT (identity),
    so a plain string would make every replayed case spuriously fail.
    """
    corpus = FileRegressionCorpus(tmp_path)
    corpus.append(_case(), sponsor="alice")

    fresh = FileRegressionCorpus(tmp_path)
    loaded = fresh.cases()
    assert len(loaded) == 1
    expect = loaded[0].containment_case.expect
    # Identity check — must be the enum member, not a plain string
    assert expect is Containment.HALTED


def test_nested_dataclasses_are_instances_not_dicts(tmp_path):
    """SafetyProperty and ContainmentCase must be dataclass instances after
    reload so _case_digest_payload (which calls dataclasses.asdict on them)
    does not TypeError.
    """
    import dataclasses

    corpus = FileRegressionCorpus(tmp_path)
    corpus.append(_case(), sponsor="alice")

    fresh = FileRegressionCorpus(tmp_path)
    case = fresh.cases()[0]
    # These must NOT raise TypeError — they would if the fields were plain dicts
    dataclasses.asdict(case.property)
    dataclasses.asdict(case.containment_case)


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------


def test_corrupt_case_hash_raises_corpus_integrity_error(tmp_path):
    corpus = FileRegressionCorpus(tmp_path)
    corpus.append(_case(), sponsor="alice")

    case_file = tmp_path / "case-00001.json"
    with open(case_file) as f:
        data = json.load(f)
    data["case_hash"] = "sha256:" + "ff" * 32  # wrong hash
    with open(case_file, "w") as f:
        json.dump(data, f)

    fresh = FileRegressionCorpus(tmp_path)
    with pytest.raises(CorpusIntegrityError):
        fresh.verify()


def test_corrupt_predecessor_hash_raises_corpus_integrity_error(tmp_path):
    corpus = FileRegressionCorpus(tmp_path)
    c1 = _case()
    corpus.append(c1, sponsor="alice")
    c2 = _case(case_id="c-002", predecessor=c1.case_hash)
    corpus.append(c2, sponsor="bob")

    # Corrupt c1's hash without updating c2's predecessor — chain break
    case_file = tmp_path / "case-00001.json"
    with open(case_file) as f:
        data = json.load(f)
    data["case_hash"] = "sha256:" + "ee" * 32
    with open(case_file, "w") as f:
        json.dump(data, f)

    fresh = FileRegressionCorpus(tmp_path)
    with pytest.raises(CorpusIntegrityError):
        fresh.verify()


def test_corrupt_verdict_string_raises_corpus_integrity_error_not_valueerror(tmp_path):
    """A bad Containment value in a case file must surface as CorpusIntegrityError,
    not as a raw ValueError from Containment(bad_string).

    verify() must catch ValueError/KeyError from deserialization and re-raise
    as CorpusIntegrityError with a legible message.
    """
    corpus = FileRegressionCorpus(tmp_path)
    corpus.append(_case(), sponsor="alice")

    case_file = tmp_path / "case-00001.json"
    with open(case_file) as f:
        data = json.load(f)
    data["containment_case"]["expect"] = "not_a_valid_containment_value"
    with open(case_file, "w") as f:
        json.dump(data, f)

    fresh = FileRegressionCorpus(tmp_path)
    with pytest.raises(CorpusIntegrityError):
        fresh.verify()


# ---------------------------------------------------------------------------
# Sponsor round-trip
# ---------------------------------------------------------------------------


def test_sponsor_attribution_round_trips(tmp_path):
    corpus = FileRegressionCorpus(tmp_path)
    c1 = _case()
    corpus.append(c1, sponsor="alice.reviewer")
    c2 = _case(case_id="c-002", predecessor=c1.case_hash)
    corpus.append(c2, sponsor="bob.reviewer")

    fresh = FileRegressionCorpus(tmp_path)
    assert fresh.sponsors() == ["alice.reviewer", "bob.reviewer"]


# ---------------------------------------------------------------------------
# Append validation (mirrors InMemoryRegressionCorpus guards)
# ---------------------------------------------------------------------------


def test_append_rejects_blank_sponsor(tmp_path):
    corpus = FileRegressionCorpus(tmp_path)
    with pytest.raises(RatchetError, match="sponsor"):
        corpus.append(_case(), sponsor="   ")


def test_append_rejects_wrong_predecessor(tmp_path):
    corpus = FileRegressionCorpus(tmp_path)
    corpus.append(_case(), sponsor="alice")
    stale = _case(case_id="c-002", predecessor="genesis")  # tail is c-001 now
    with pytest.raises(RatchetError, match="predecessor"):
        corpus.append(stale, sponsor="alice")


def test_append_rejects_tampered_hash(tmp_path):
    import dataclasses as dc

    corpus = FileRegressionCorpus(tmp_path)
    forged = dc.replace(_case(), case_hash="sha256:" + "00" * 32)
    with pytest.raises(RatchetError, match="hash"):
        corpus.append(forged, sponsor="alice")


# ---------------------------------------------------------------------------
# End-to-end: admit() against a FileRegressionCorpus
# ---------------------------------------------------------------------------


async def test_admit_identity_manifest_against_file_corpus(tmp_path):
    """End-to-end proof that enum reconstruction makes admit()'s IS NOT
    comparison work correctly.

    An identity manifest must be admitted by a FileRegressionCorpus that was
    seeded in one process and reloaded in another (simulated by a fresh instance
    over the same tmp_path).  If the Containment enum were reconstructed as a
    plain string, every containment replay would spuriously fail and verdict
    would be admitted=False at the corpus layer.
    """
    from pydantic import BaseModel

    from drawbore import Pipeline, agent
    from drawbore.config import AgentCatalog, to_config
    from drawbore.escalation import HasConfidence
    from drawbore.pipeline.binding import From
    from drawbore.ratchet import FileRegressionCorpus, admit, derive_cases
    from drawbore.tools import ToolRegistry

    FETCH = "data:fetch_fc"

    class TxIn(BaseModel):
        transaction_id: str

    class TxRow(BaseModel):
        amount: int
        status: str

    class RiskIn(BaseModel):
        amount: int
        status: str

    class RiskScore(BaseModel, HasConfidence):
        risk_level: str
        confidence: float
        factors: list[str]

    retriever_fc = agent(
        name="retriever_fc", input=TxIn, output=TxRow, tools=[FETCH],
    )(lambda v, tools: None)

    # Override with a real async handler via module-level definition pattern
    @agent(name="retriever_fc2", input=TxIn, output=TxRow, tools=[FETCH])
    async def retriever_fc2(v: TxIn, tools) -> TxRow:
        row = await tools.call(FETCH, {"id": v.transaction_id})
        return TxRow(**row)

    @agent(name="risk_scorer_fc", input=RiskIn, output=RiskScore, model="risk-1.0")
    async def risk_scorer_fc(v: RiskIn) -> RiskScore: ...

    reg = ToolRegistry()

    async def _fetch(args):
        return {"amount": 100, "status": "pending"}

    reg.register_tool(FETCH, _fetch, allowed_operations=("invoke",), schema={"type": "object"})

    pipeline = Pipeline(
        "file-corpus-demo", version="1.0.0", registry=reg, confidence_threshold=0.8
    )
    pipeline.add(retriever_fc2)
    pipeline.add(
        risk_scorer_fc,
        inputs={
            "amount": From("retriever_fc2.amount"),
            "status": From("retriever_fc2.status"),
        },
        depends_on=["retriever_fc2"],
    )

    catalog = AgentCatalog()
    catalog.register("demo/retriever_fc2", retriever_fc2)
    catalog.register("demo/risk_scorer_fc", risk_scorer_fc)

    MOCKS = {
        "mock_tools": {FETCH: {"amount": 100, "status": "pending"}},
        "mock_model_responses": {
            "risk_scorer_fc": {
                "risk_level": "low",
                "confidence": 0.95,
                "factors": ["clean"],
            },
        },
    }
    INITIAL = TxIn(transaction_id="t42")

    config = to_config(pipeline, agents=catalog)
    async with pipeline.test_mode(**MOCKS) as tp:
        result = await tp.run(INITIAL)

    # Seed into a FileRegressionCorpus
    write_corpus = FileRegressionCorpus(tmp_path)
    for case in derive_cases(
        config, pipeline, result,
        initial=INITIAL,
        baseline_mocks=MOCKS,
        derived_at="2026-06-13T00:00:00Z",
        predecessor=None,
    ):
        write_corpus.append(case, sponsor="alice.reviewer")

    assert write_corpus.cases(), "corpus must be non-empty to proceed"

    # Reload from disk — simulates a process restart
    read_corpus = FileRegressionCorpus(tmp_path)
    read_corpus.verify()  # chain must be intact after reload

    # Identity manifest must ADMIT against the reloaded file corpus.
    # This requires Containment enum reconstruction: admit() uses
    #   ``observed is not case.containment_case.expect``
    # A plain string would never be the same object, so every case would
    # spuriously fail and admit() would return admitted=False.
    candidate = config.model_dump(mode="json")
    verdict = await admit(
        candidate,
        agents=catalog,
        corpus=read_corpus,
        sponsor="alice.reviewer",
        baseline_config=config,
        initial_input=INITIAL,
        baseline_mocks=MOCKS,
        derived_at="2026-06-13T01:00:00Z",
        registry=reg,
    )
    assert verdict.admitted is True, (
        f"Identity manifest must admit against reloaded FileRegressionCorpus; "
        f"got rejection_layer={verdict.rejection_layer!r} reason={verdict.reason!r}"
    )
    assert verdict.pipeline is not None
