import pytest

from drawbore.evidence import (
    EvidenceStore, InMemoryEvidenceStore, EvidenceHandle, EvidenceRetrievalError,
)


def _handle(handle_id="h1"):
    return EvidenceHandle(
        handle_id=handle_id, run_id="r1", step=0, source_agent="screen",
        content_type="json_rows", original_hash="o" * 16, compressed_hash="c" * 16,
        original_tokens=4000, compressed_tokens=900, transform="json_rows",
    )


ORIGINAL = {"rows": [{"id": i, "amount": i * 10} for i in range(100)]}
COMPRESSED = {"rows": [{"id": 0, "amount": 0}, {"id": 99, "amount": 990}]}


def test_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        EvidenceStore()


def test_store_and_metadata_does_not_return_content():
    store = InMemoryEvidenceStore()
    store.put(_handle(), original=ORIGINAL, compressed=COMPRESSED)
    meta = store.metadata("h1")
    assert meta.handle_id == "h1"
    assert meta.original_tokens == 4000
    assert not hasattr(meta, "rows")


def test_retrieve_full_returns_the_original():
    store = InMemoryEvidenceStore()
    store.put(_handle(), original=ORIGINAL, compressed=COMPRESSED)
    assert store.retrieve_full("h1") == ORIGINAL


def test_search_returns_bounded_matches_from_the_original():
    store = InMemoryEvidenceStore()
    store.put(_handle(), original=ORIGINAL, compressed=COMPRESSED)
    matches = store.search("h1", query="990", max_results=10)
    assert any("990" in str(m) for m in matches)
    assert len(store.search("h1", query="amount", max_results=3)) <= 3


def test_search_indexes_scalar_dict_values_not_just_lists():
    # A dict carrying both a record list and a scalar field: a match in the scalar
    # field must still be findable (nothing in the dict is silently unsearchable).
    store = InMemoryEvidenceStore()
    store.put(
        _handle(),
        original={"summary": "out of memory", "rows": [{"id": 1}]},
        compressed=COMPRESSED,
    )
    assert store.search("h1", query="out of memory", max_results=10)


def test_search_max_results_zero_or_negative_returns_empty():
    store = InMemoryEvidenceStore()
    store.put(_handle(), original=ORIGINAL, compressed=COMPRESSED)
    assert store.search("h1", query="amount", max_results=0) == []
    assert store.search("h1", query="amount", max_results=-5) == []


def test_unknown_handle_fails_closed():
    store = InMemoryEvidenceStore()
    with pytest.raises(EvidenceRetrievalError):
        store.retrieve_full("nope")
    with pytest.raises(EvidenceRetrievalError):
        store.metadata("nope")
    with pytest.raises(EvidenceRetrievalError):
        store.search("nope", query="x", max_results=5)


def test_expiry_fails_closed():
    clock = {"t": 1000.0}
    store = InMemoryEvidenceStore(clock=lambda: clock["t"])
    store.put(_handle(), original=ORIGINAL, compressed=COMPRESSED, ttl_seconds=60)
    assert store.retrieve_full("h1") == ORIGINAL
    clock["t"] = 1060.0                              # exactly at ttl: still live (closed interval)
    assert store.retrieve_full("h1") == ORIGINAL
    clock["t"] = 1100.0                              # past ttl
    with pytest.raises(EvidenceRetrievalError):
        store.retrieve_full("h1")


def test_no_ttl_never_expires():
    clock = {"t": 0.0}
    store = InMemoryEvidenceStore(clock=lambda: clock["t"])
    store.put(_handle(), original=ORIGINAL, compressed=COMPRESSED, ttl_seconds=None)
    clock["t"] = 10_000_000.0
    assert store.retrieve_full("h1") == ORIGINAL
