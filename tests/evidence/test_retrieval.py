import pytest

from drawbore.tools import (
    ToolRegistry, ToolProxy, TokenIssuer, RunContext, build_tool_context,
    ToolAccessError, set_run_context, reset_run_context,
)
from drawbore.evidence import (
    EvidencePolicy, InMemoryEvidenceStore, EvidenceHandle, EvidenceRetrievalError,
    register_evidence_tool, EVIDENCE_TOOL_REF,
)


def _stored(store, *, allow_full=False, allow_search=True):
    handle = EvidenceHandle(
        handle_id="h1", run_id="r1", step=0, source_agent="screen",
        content_type="json_rows", original_hash="o" * 16, compressed_hash="c" * 16,
        original_tokens=4000, compressed_tokens=900, transform="json_rows",
    )
    original = {"records": [{"id": i, "amount": i * 10} for i in range(50)]}
    store.put(handle, original=original, compressed={"records": original["records"][:2]})
    store.set_policy("h1", allow_full=allow_full, allow_search=allow_search)
    return handle, original


async def _call(proxy, issuer, args):
    ctx = build_tool_context([EVIDENCE_TOOL_REF], proxy, issuer)
    token = set_run_context(RunContext(run_id="r1", step=0))
    try:
        return await ctx.call(EVIDENCE_TOOL_REF, args)
    finally:
        reset_run_context(token)


async def test_search_retrieval_through_the_proxy_with_a_jit_token():
    registry = ToolRegistry()
    store = InMemoryEvidenceStore()
    _stored(store, allow_search=True)
    register_evidence_tool(registry, store=store)
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer)
    out = await _call(proxy, issuer, {"handle_id": "h1", "mode": "search", "query": "490"})
    assert any("490" in str(m) for m in out)


async def test_full_retrieval_denied_by_policy_fails_closed():
    registry = ToolRegistry()
    store = InMemoryEvidenceStore()
    _stored(store, allow_full=False)
    register_evidence_tool(registry, store=store)
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer)
    with pytest.raises(EvidenceRetrievalError):
        await _call(proxy, issuer, {"handle_id": "h1", "mode": "full"})


async def test_search_retrieval_denied_by_policy_fails_closed():
    registry = ToolRegistry()
    store = InMemoryEvidenceStore()
    _stored(store, allow_full=False, allow_search=False)
    register_evidence_tool(registry, store=store)
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer)
    with pytest.raises(EvidenceRetrievalError):
        await _call(proxy, issuer, {"handle_id": "h1", "mode": "search", "query": "x"})


async def test_full_retrieval_allowed_returns_the_original():
    registry = ToolRegistry()
    store = InMemoryEvidenceStore()
    _, original = _stored(store, allow_full=True)
    register_evidence_tool(registry, store=store)
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer)
    out = await _call(proxy, issuer, {"handle_id": "h1", "mode": "full"})
    assert out == original


async def test_unknown_handle_fails_closed():
    registry = ToolRegistry()
    store = InMemoryEvidenceStore()
    register_evidence_tool(registry, store=store)
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer)
    with pytest.raises(EvidenceRetrievalError):
        await _call(proxy, issuer, {"handle_id": "ghost", "mode": "search", "query": "x"})


async def test_unknown_mode_fails_closed():
    registry = ToolRegistry()
    store = InMemoryEvidenceStore()
    _stored(store, allow_search=True)
    register_evidence_tool(registry, store=store)
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer)
    with pytest.raises(EvidenceRetrievalError):
        await _call(proxy, issuer, {"handle_id": "h1", "mode": "exfiltrate"})


async def test_bad_max_results_fails_as_evidence_error_not_value_error():
    # A non-integer model-supplied max_results escalates legibly (evidence_error),
    # not as a raw ValueError that the pipeline would mislabel agent_error.
    registry = ToolRegistry()
    store = InMemoryEvidenceStore()
    _stored(store, allow_search=True)
    register_evidence_tool(registry, store=store)
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer)
    with pytest.raises(EvidenceRetrievalError):
        await _call(proxy, issuer, {"handle_id": "h1", "mode": "search", "query": "x", "max_results": "lots"})


async def test_search_is_capped_and_cannot_reconstruct_the_full_original():
    # Search is SCOPED: an empty query + a huge max_results must NOT return the whole
    # original (that would defeat allow_full=False). The cap bounds the result count.
    registry = ToolRegistry()
    store = InMemoryEvidenceStore()
    _, original = _stored(store, allow_full=False, allow_search=True)
    register_evidence_tool(registry, store=store)
    # widen the original well past the cap
    big = {"records": [{"id": i} for i in range(1000)]}
    store.put(
        EvidenceHandle(
            handle_id="big", run_id="r1", step=0, source_agent="screen",
            content_type="json_rows", original_hash="o" * 16, compressed_hash="c" * 16,
            original_tokens=9999, compressed_tokens=10, transform="json_rows",
        ),
        original=big, compressed={"records": []},
    )
    store.set_policy("big", allow_full=False, allow_search=True)
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer)
    out = await _call(proxy, issuer, {"handle_id": "big", "mode": "search", "query": "", "max_results": 100000})
    assert len(out) <= 100              # bounded — not the full 1000-row original
    assert len(out) < len(big["records"])


async def test_handle_visibility_alone_does_not_authorize_retrieval():
    # An agent that did NOT declare the evidence tool cannot reach it, even with a
    # valid handle id — the proxy/ToolContext is authoritative, the handle is not.
    registry = ToolRegistry()
    store = InMemoryEvidenceStore()
    _stored(store, allow_search=True)
    register_evidence_tool(registry, store=store)
    issuer = TokenIssuer()
    proxy = ToolProxy(registry, issuer)
    ctx = build_tool_context([], proxy, issuer)   # declares NO tools
    token = set_run_context(RunContext(run_id="r1", step=0))
    try:
        with pytest.raises(ToolAccessError):
            await ctx.call(EVIDENCE_TOOL_REF, {"handle_id": "h1", "mode": "search", "query": "x"})
    finally:
        reset_run_context(token)
