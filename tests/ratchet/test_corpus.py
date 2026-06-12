# tests/ratchet/test_corpus.py
import dataclasses

import pytest

from drawbore.ratchet import (
    CorpusIntegrityError,
    InMemoryRegressionCorpus,
    RatchetError,
    RegressionCase,
    SafetyProperty,
    mocks_fingerprint,
    seal_case,
)
from drawbore.testing import Containment, ContainmentCase

PROP = SafetyProperty(
    dim="run_status", step="risk_scorer", assertion="neq", value="completed",
    description="a below-threshold confidence from 'risk_scorer' must not complete the run",
)
CC = ContainmentCase(
    name="low_confidence:risk_scorer", kind="low_confidence", target="risk_scorer",
    payload={"risk_level": "low", "confidence": 0.0, "factors": []},
    expect=Containment.HALTED,
)


def _case(case_id="c-001", predecessor="genesis"):
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


def test_seal_case_is_deterministic_and_sets_hash():
    a, b = _case(), _case()
    assert a.case_hash == b.case_hash
    assert a.case_hash.startswith("sha256:")
    assert a.predecessor_hash == "genesis"


def test_seal_case_hash_changes_with_any_field():
    a = _case()
    b = _case(case_id="c-002")
    assert a.case_hash != b.case_hash


def test_empty_corpus_root_is_genesis_and_verifies():
    corpus = InMemoryRegressionCorpus()
    assert corpus.root() == "genesis"
    corpus.verify()  # no raise


def test_append_advances_root_and_chains():
    corpus = InMemoryRegressionCorpus()
    c1 = _case()
    corpus.append(c1, sponsor="a.reviewer")
    assert corpus.root() == c1.case_hash
    c2 = _case(case_id="c-002", predecessor=c1.case_hash)
    corpus.append(c2, sponsor="a.reviewer")
    assert corpus.root() == c2.case_hash
    assert [c.case_id for c in corpus.cases()] == ["c-001", "c-002"]
    corpus.verify()  # no raise
    assert corpus.sponsors() == ["a.reviewer", "a.reviewer"]


def test_append_rejects_blank_sponsor():
    corpus = InMemoryRegressionCorpus()
    with pytest.raises(RatchetError, match="sponsor"):
        corpus.append(_case(), sponsor="   ")


def test_append_rejects_wrong_predecessor():
    corpus = InMemoryRegressionCorpus()
    corpus.append(_case(), sponsor="a.reviewer")
    stale = _case(case_id="c-002", predecessor="genesis")  # chain tail is c-001 now
    with pytest.raises(RatchetError, match="predecessor"):
        corpus.append(stale, sponsor="a.reviewer")


def test_append_rejects_tampered_hash():
    corpus = InMemoryRegressionCorpus()
    forged = dataclasses.replace(_case(), case_hash="sha256:" + "00" * 32)
    with pytest.raises(RatchetError, match="hash"):
        corpus.append(forged, sponsor="a.reviewer")


def test_verify_fails_closed_on_in_place_tamper():
    corpus = InMemoryRegressionCorpus()
    corpus.append(_case(), sponsor="a.reviewer")
    # simulate storage tamper: swap the stored case for one with an edited field
    tampered = dataclasses.replace(corpus.cases()[0], derived_at="1999-01-01T00:00:00Z")
    corpus._cases[0] = tampered
    with pytest.raises(CorpusIntegrityError):
        corpus.verify()


def test_mocks_fingerprint_is_two_key_pinned():
    a = mocks_fingerprint({"mock_tools": {"t": {"ok": True}}})
    b = mocks_fingerprint({"mock_tools": {"t": {"ok": True}}, "mock_model_responses": {}})
    assert a == b                      # absent key == empty key
    with pytest.raises(RatchetError, match="mock_loop_scripts"):
        mocks_fingerprint({"mock_loop_scripts": {"a": []}})
