import pytest

from drawbore.evidence import (
    EvidencePolicy, InMemoryEvidenceStore, EvidenceStoreError, compress_for_model,
)


def _ctx(**kw):
    base = dict(run_id="r1", step=0, source_agent="screen")
    base.update(kw)
    return base


def _rows(n):
    return {"records": [{"id": i, "amount": i, "status": "ok"} for i in range(n)]}


def test_disabled_policy_is_byte_identical_passthrough():
    store = InMemoryEvidenceStore()
    payload = _rows(500)
    view, decision, handle = compress_for_model(
        payload, EvidencePolicy(enabled=False), store=store, **_ctx())
    import json
    assert json.dumps(view, sort_keys=True) == json.dumps(payload, sort_keys=True)
    assert decision.decision == "passthrough"
    assert handle is None
    with pytest.raises(Exception):
        store.metadata(decision.handle_id or "none")


def test_below_min_tokens_is_passthrough_with_decision():
    store = InMemoryEvidenceStore()
    payload = _rows(2)
    view, decision, handle = compress_for_model(
        payload, EvidencePolicy(enabled=True, min_tokens=10_000_000), store=store, **_ctx())
    assert view == payload
    assert decision.decision == "passthrough"
    assert "min_tokens" in decision.reason
    assert handle is None


def test_simulate_records_decision_without_changing_payload():
    store = InMemoryEvidenceStore()
    payload = _rows(500)
    view, decision, handle = compress_for_model(
        payload, EvidencePolicy(enabled=True, mode="simulate"), store=store, **_ctx())
    assert view == payload
    assert decision.decision == "compressed"
    assert decision.compressed_tokens is not None and decision.compressed_tokens < decision.original_tokens
    assert "simulate" in decision.reason


def test_compress_shrinks_view_stores_original_and_builds_handle():
    store = InMemoryEvidenceStore()
    payload = _rows(500)
    view, decision, handle = compress_for_model(
        payload, EvidencePolicy(enabled=True, mode="compress"), store=store, **_ctx())
    assert len(view["records"]) < 500
    assert decision.decision == "compressed"
    assert handle is not None
    assert store.retrieve_full(handle.handle_id) == payload
    assert store.metadata(handle.handle_id).transform == "json_rows"


def test_handle_id_is_deterministic_for_same_input_and_policy():
    payload = _rows(500)
    p = EvidencePolicy(enabled=True)
    _, _, h1 = compress_for_model(payload, p, store=InMemoryEvidenceStore(), **_ctx())
    _, _, h2 = compress_for_model(payload, p, store=InMemoryEvidenceStore(), **_ctx())
    assert h1.handle_id == h2.handle_id


def test_no_applicable_transform_is_passthrough():
    store = InMemoryEvidenceStore()
    payload = {"summary": "x" * 5000}
    view, decision, handle = compress_for_model(
        payload, EvidencePolicy(enabled=True), store=store, **_ctx())
    assert view == payload
    assert decision.decision == "passthrough"
    assert "no applicable transform" in decision.reason


def test_store_failure_fails_closed_when_original_required():
    class _BrokenStore(InMemoryEvidenceStore):
        def put(self, *a, **k):
            raise EvidenceStoreError("disk full")
    payload = _rows(500)
    with pytest.raises(EvidenceStoreError):
        compress_for_model(
            payload, EvidencePolicy(enabled=True, require_original_store=True),
            store=_BrokenStore(), **_ctx())


def test_store_failure_soft_fails_to_passthrough_when_original_not_required():
    # When the original is NOT required, a store failure must return the ORIGINAL
    # payload (passthrough) — never the compressed view without a retained original.
    class _BrokenStore(InMemoryEvidenceStore):
        def put(self, *a, **k):
            raise EvidenceStoreError("disk full")
    payload = _rows(500)
    view, decision, handle = compress_for_model(
        payload,
        EvidencePolicy(enabled=True, mode="compress", require_original_store=False),
        store=_BrokenStore(), **_ctx())
    assert view == payload                 # the original, NOT the compressed view
    assert decision.decision == "passthrough"
    assert handle is None
