"""The evidence retrieval tool.

Retrieval is a declared, proxy-backed Drawbore tool — a handle is data, not
authority: the proxy + single-use JIT token are authoritative.
``register_evidence_tool`` registers a builtin tool whose handler serves
full/search retrieval from the store, gated by the per-handle policy flags and
expiry, failing closed (``EvidenceRetrievalError``). The registry is duck-typed
(mirrors ``register_mcp_server``) so ``drawbore.evidence`` never imports
``drawbore.tools``.
"""

from __future__ import annotations

from typing import Any

from .errors import EvidenceRetrievalError
from .store import EvidenceStore

EVIDENCE_TOOL_REF = "evidence://retrieve"
_DEFAULT_MAX_RESULTS = 20
# Search is SCOPED retrieval, not wholesale: a hard ceiling keeps it from
# reconstructing the whole original (which would defeat allow_full=False).
_MAX_SEARCH_RESULTS = 100


def register_evidence_tool(registry, *, store: EvidenceStore, ref: str = EVIDENCE_TOOL_REF) -> str:
    """Register the evidence retrieval tool into ``registry`` (``kind="builtin"``).
    Returns the tool ref. The handler routes to ``store`` and enforces per-handle
    retrieval policy + expiry; the proxy enforces declaration/JIT/scope/breaker."""

    async def handler(args: Any) -> Any:
        if not isinstance(args, dict):
            raise EvidenceRetrievalError("evidence retrieval requires an args object")
        handle_id = args.get("handle_id")
        mode = args.get("mode", "search")
        if not handle_id:
            raise EvidenceRetrievalError("evidence retrieval requires 'handle_id'")
        allow_full, allow_search = store.policy_for(handle_id)  # fails closed if absent/expired
        if mode == "full":
            if not allow_full:
                raise EvidenceRetrievalError(
                    f"full retrieval of '{handle_id}' is not allowed by policy"
                )
            return store.retrieve_full(handle_id)
        if mode == "search":
            if not allow_search:
                raise EvidenceRetrievalError(
                    f"search retrieval of '{handle_id}' is not allowed by policy"
                )
            query = args.get("query", "")
            raw = args.get("max_results", _DEFAULT_MAX_RESULTS)
            try:
                requested = int(raw)
            except (TypeError, ValueError):
                # A bad model-supplied arg is an evidence failure (legible halt),
                # not a generic agent_error from an uncaught ValueError.
                raise EvidenceRetrievalError(f"'max_results' must be an integer, got {raw!r}")
            # Cap so search stays scoped — it must not reconstruct the full original.
            max_results = min(max(0, requested), _MAX_SEARCH_RESULTS)
            return store.search(handle_id, query=query, max_results=max_results)
        raise EvidenceRetrievalError(f"unknown retrieval mode '{mode}' (use 'search' or 'full')")

    handler.__name__ = "evidence::retrieve"
    registry.register_builtin(
        ref, handler,
        schema={
            "type": "object",
            "properties": {
                "handle_id": {"type": "string"},
                "mode": {"type": "string", "enum": ["search", "full"]},
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["handle_id"],
        },
    )
    return ref
