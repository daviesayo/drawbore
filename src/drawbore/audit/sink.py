"""Append-only audit sink.

``AuditSink`` is the write seam; ``InMemoryAuditSink`` is the in-process default
(mirrors ``RecordingDispatcher``/the in-memory ``CheckpointStore``). Durable,
queryable backends implement the same ABC. "Append-only" is the contract: a sink
only appends records and never mutates or deletes them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .records import AuditRecord


class AuditSink(ABC):
    """Write-only destination for audit records."""

    @abstractmethod
    def write(self, record: AuditRecord) -> None:
        """Append ``record``. Implementations MUST NOT mutate or drop prior
        records (append-only contract)."""


class InMemoryAuditSink(AuditSink):
    """In-process append-only sink. ``records`` returns a copy so callers cannot
    mutate the history through the view; ``by_run_id`` is the basic run-id query."""

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    def write(self, record: AuditRecord) -> None:
        self._records.append(record)

    @property
    def records(self) -> list[AuditRecord]:
        return list(self._records)

    def by_run_id(self, run_id: str) -> AuditRecord | None:
        for record in self._records:
            if record.run_id == run_id:
                return record
        return None
