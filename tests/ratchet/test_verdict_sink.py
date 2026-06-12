# tests/ratchet/test_verdict_sink.py
from drawbore.ratchet import InMemoryRatchetSink, RatchetVerdict


def _verdict(**over):
    base = dict(
        admitted=False,
        pipeline=None,
        manifest_fingerprint="sha256:" + "ab" * 32,
        rejection_layer="corpus",
        authority_diff=None,
        failed_case_id="c-002",
        failed_property_description="a below-threshold confidence from 'risk_scorer' must not complete the run",
        corpus_root_before="sha256:" + "9f" * 32,
        corpus_root_after=None,
    )
    base.update(over)
    return RatchetVerdict(**base)


def test_rejected_verdict_legible_names_the_case_and_keeps_root():
    text = _verdict().legible()
    assert "REJECTED" in text
    assert "c-002" in text
    assert "must not complete the run" in text
    assert ("sha256:" + "9f" * 32) in text          # root named
    assert "unchanged" in text.lower()


def test_admitted_verdict_legible_names_roots():
    text = _verdict(
        admitted=True, rejection_layer=None, failed_case_id=None,
        failed_property_description=None,
        corpus_root_after="sha256:" + "3c" * 32,
    ).legible()
    assert "ADMITTED" in text
    assert ("sha256:" + "3c" * 32) in text


def test_admitted_verdict_with_no_growth_says_unchanged():
    text = _verdict(
        admitted=True, rejection_layer=None, failed_case_id=None,
        failed_property_description=None, corpus_root_after=None,
    ).legible()
    assert "ADMITTED" in text
    assert "unchanged" in text.lower()
    assert ("sha256:" + "9f" * 32) in text
    assert "no new cases derived" in text


def test_sink_is_append_only_and_returns_copies():
    sink = InMemoryRatchetSink()
    v1, v2 = _verdict(), _verdict(admitted=True, rejection_layer=None)
    sink.write(v1)
    sink.write(v2)
    got = sink.verdicts
    assert got == [v1, v2]
    got.clear()                       # mutating the view must not touch history
    assert sink.verdicts == [v1, v2]
