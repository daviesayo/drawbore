"""Evidence store — originals retained, policy-gated retrieval, expiry.

``EvidenceStore`` is the ABC; ``InMemoryEvidenceStore`` is the in-process default
(mirrors ``CheckpointStore``/``AuditSink`` — durable stores are managed-service).
The store keeps both the original and the compressed view keyed by a deterministic
``handle_id`` and serves: metadata (the handle, no content), full retrieval, and
bounded search within the original. It FAILS CLOSED on an unknown or expired
handle. It enforces existence/expiry only; *policy* gating (full vs
search, scope) is the retrieval tool's job (later task) — the store is the substrate.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from .errors import EvidenceRetrievalError
from .records import EvidenceHandle


class EvidenceStore(ABC):
    @abstractmethod
    def put(
        self, handle: EvidenceHandle, *, original: Any, compressed: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store the original + compressed forms under ``handle.handle_id``."""

    @abstractmethod
    def metadata(self, handle_id: str) -> EvidenceHandle:
        """Return the handle (metadata only, no content). Fail closed if absent."""

    @abstractmethod
    def retrieve_full(self, handle_id: str) -> Any:
        """Return the stored original. Fail closed if absent/expired."""

    @abstractmethod
    def search(self, handle_id: str, *, query: str, max_results: int) -> list[Any]:
        """Return up to ``max_results`` items of the original that match ``query``
        (deterministic substring match). Fail closed if absent/expired."""

    @abstractmethod
    def set_policy(self, handle_id: str, *, allow_full: bool, allow_search: bool) -> None:
        """Record per-handle retrieval permissions. Fail closed if absent."""

    @abstractmethod
    def policy_for(self, handle_id: str) -> tuple[bool, bool]:
        """Return ``(allow_full, allow_search)`` for a live handle (fail closed if
        absent/expired)."""

    @abstractmethod
    def delete(self, handle_id: str) -> None:
        """Evict a handle (idempotent — absent is not an error). Used to drop an
        entry whose compressed view was rejected, so the store never disagrees with
        the audit trail (the run recorded passthrough, no handle)."""


@dataclass
class _Entry:
    handle: EvidenceHandle
    original: Any
    compressed: Any
    stored_at: float
    ttl_seconds: int | None
    allow_full: bool = False
    allow_search: bool = True


class InMemoryEvidenceStore(EvidenceStore):
    """In-process evidence store. ``clock`` is injectable for deterministic expiry
    tests (defaults to ``time.monotonic``, like ``ToolProxy``)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._entries: dict[str, _Entry] = {}
        self._clock = clock

    def put(
        self, handle: EvidenceHandle, *, original: Any, compressed: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        self._entries[handle.handle_id] = _Entry(
            handle=handle, original=original, compressed=compressed,
            stored_at=self._clock(), ttl_seconds=ttl_seconds,
        )

    def _live_entry(self, handle_id: str) -> _Entry:
        entry = self._entries.get(handle_id)
        if entry is None:
            raise EvidenceRetrievalError(f"unknown evidence handle '{handle_id}'")
        if entry.ttl_seconds is not None and (self._clock() - entry.stored_at) > entry.ttl_seconds:
            raise EvidenceRetrievalError(f"evidence handle '{handle_id}' has expired")
        return entry

    def metadata(self, handle_id: str) -> EvidenceHandle:
        return self._live_entry(handle_id).handle

    def retrieve_full(self, handle_id: str) -> Any:
        return self._live_entry(handle_id).original

    def search(self, handle_id: str, *, query: str, max_results: int) -> list[Any]:
        """Return up to ``max_results`` items of the original matching ``query``.
        ``max_results <= 0`` yields ``[]`` (the negative slice bound is clamped to 0)."""
        original = self._live_entry(handle_id).original
        items = _iter_rows(original)
        matches = [item for item in items if query in _searchable(item)]
        return matches[: max(0, max_results)]

    def set_policy(self, handle_id: str, *, allow_full: bool, allow_search: bool) -> None:
        """Record per-handle retrieval permissions (set by compress wiring/tests)."""
        entry = self._entries.get(handle_id)
        if entry is None:
            raise EvidenceRetrievalError(f"unknown evidence handle '{handle_id}'")
        entry.allow_full = allow_full
        entry.allow_search = allow_search

    def policy_for(self, handle_id: str) -> tuple[bool, bool]:
        """Return ``(allow_full, allow_search)`` for a live handle (fail closed if absent/expired)."""
        entry = self._live_entry(handle_id)
        return entry.allow_full, entry.allow_search

    def delete(self, handle_id: str) -> None:
        """Evict a handle (idempotent — absent is not an error)."""
        self._entries.pop(handle_id, None)


def _iter_rows(original: Any) -> list[Any]:
    """The searchable items of an original: a list yields its items; a dict yields
    each value (list-valued entries flattened, scalar values kept as single items —
    so nothing in the dict is silently unsearchable); anything else is one item.
    Deterministic order (insertion / original order)."""
    if isinstance(original, list):
        return list(original)
    if isinstance(original, dict):
        rows: list[Any] = []
        for value in original.values():
            if isinstance(value, list):
                rows.extend(value)
            else:
                rows.append(value)
        if rows:
            return rows
    return [original]


def _searchable(item: Any) -> str:
    import json
    return json.dumps(item, sort_keys=True, default=str)
