import dataclasses

import pytest

from drawbore.evidence import EvidenceDecision, EvidenceHandle


def _handle():
    return EvidenceHandle(
        handle_id="h1", run_id="r1", step=0, source_agent="screen",
        content_type="json_rows", original_hash="o" * 16, compressed_hash="c" * 16,
        original_tokens=4000, compressed_tokens=900, transform="json_rows",
    )


def test_handle_is_immutable():
    h = _handle()
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.handle_id = "x"


def test_decision_is_immutable_and_renders_legibly():
    d = EvidenceDecision(
        decision="compressed", reason="rows above threshold", policy="aml",
        transform="json_rows", original_tokens=4000, compressed_tokens=900,
        handle_id="h1", warnings=("dropped 1100 rows",),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.decision = "x"
    text = d.legible()
    assert "compressed" in text
    assert "json_rows" in text
    assert "4000" in text and "900" in text
    assert "h1" in text
    assert "dropped 1100 rows" in text


def test_passthrough_decision_renders_without_handle_or_tokens():
    d = EvidenceDecision(
        decision="passthrough", reason="below min_tokens", policy="aml",
        transform=None, original_tokens=120, compressed_tokens=None, handle_id=None,
    )
    text = d.legible()
    assert "passthrough" in text
    assert "below min_tokens" in text
