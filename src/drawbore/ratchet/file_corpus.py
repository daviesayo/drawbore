"""File-backed regression corpus for cross-restart persistence.

``FileRegressionCorpus`` persists the hash-chained regression corpus to a
directory so a production ratchet's corpus survives a process restart.

File layout
-----------
One ``case-NNNNN.json`` per appended case (zero-padded 5-digit position index,
matching append order — exactly as ``FileCheckpointStore`` uses
``step-NNNNN.json``). The naming scheme is load-bearing: ``verify()``
reconstructs append order by sorting case files by filename; an unordered read
would fail to detect a chain break between out-of-order cases.

Each file contains the full serialised ``RegressionCase`` plus the per-case
``"sponsor"`` embedded inline. The in-memory corpus tracks sponsors in a
parallel position-indexed list that cannot survive a reload; embedding the
sponsor in the case record is the only design that survives a restart.

Atomic writes
-------------
Every file is written to a unique temp file and then ``os.replace``-d into
place, which is atomic on a POSIX filesystem (mirroring
``FileCheckpointStore``). A crash mid-write leaves at most a stray ``*.tmp-*``
file (ignored on read); a previously committed case is never torn.

Deserialization contract
------------------------
Two invariants are correctness-critical (a blind review caught these):

- ``ContainmentCase.expect`` MUST be reconstructed as a ``Containment`` **enum
  member** via ``Containment(raw_str)``, not left as a plain string.
  ``admit()`` compares the observed verdict against ``expect`` with **identity**
  (``observed is not case.containment_case.expect``), so a plain-string
  ``expect`` would make every replayed case spuriously fail and ``admit()``
  would reject every candidate at the corpus layer.

- The nested dataclasses (``SafetyProperty``, ``ContainmentCase``) MUST be
  rebuilt as **dataclass instances**, not left as plain dicts.
  ``_case_digest_payload`` calls ``dataclasses.asdict(...)`` on them; that
  raises ``TypeError`` on a plain dict, so ``verify()`` would fail on every
  loaded case.

``verify()`` catches ``ValueError`` / ``KeyError`` from deserialization (e.g.
``Containment(bad_string)`` or a missing field) and re-raises them as
``CorpusIntegrityError`` with a legible message — a corrupted corpus must
surface as the integrity error callers catch, never as a raw ``ValueError``.

Single-writer assumption
------------------------
Atomic rename handles a single writer safely. Concurrent multi-process appends
are **not** supported: two simultaneous appends would race on the position
index, producing duplicate ``case-NNNNN.json`` filenames or a broken chain.
Use a single process (or an external lock) when multiple writers are required.

Stdlib only; no new dependency.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from drawbore.testing import Containment, ContainmentCase

from .corpus import (
    GENESIS,
    RegressionCase,
    RegressionCorpus,
    SafetyProperty,
    _recompute_hash,
)
from .errors import CorpusIntegrityError, RatchetError


class FileRegressionCorpus(RegressionCorpus):
    """A ``RegressionCorpus`` that persists to a directory on disk.

    One ``case-NNNNN.json`` per appended case (zero-padded position index,
    matching append order). Reads are cached in memory after the first directory
    scan; ``verify()`` always reads fresh from disk for tamper detection.

    Single-writer assumption: concurrent multi-process appends are not
    supported — see module docstring.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        # Lazy in-memory cache, populated on first read.  Never used by verify().
        self._cases_cache: list[RegressionCase] | None = None
        self._sponsors_cache: list[str] | None = None

    # --- path / filename helpers -------------------------------------------

    @staticmethod
    def _case_filename(index: int) -> str:
        """Zero-padded case filename for stable lexical ordering.

        Uses a 5-digit index (1-based) matching the append position, so
        filename sort order always equals append order.
        """
        return f"case-{index:05d}.json"

    # --- atomic write -------------------------------------------------------

    def _atomic_write(self, path: Path, data: dict[str, Any]) -> None:
        """Write *data* to *path* atomically via a temp file + ``os.replace``.

        Mirrors ``FileCheckpointStore._atomic_write`` exactly: compact JSON,
        fsync before replace, unique temp name so concurrent writers never share
        a temp file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, separators=(",", ":"), sort_keys=True)
        tmp = path.with_name(
            f"{path.name}.tmp-{os.getpid()}-{os.urandom(6).hex()}"
        )
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    # --- serialisation / deserialisation ------------------------------------

    @staticmethod
    def _serialise(case: RegressionCase, sponsor: str) -> dict[str, Any]:
        """Flatten a ``RegressionCase`` to a JSON-safe dict, sponsor embedded.

        ``dataclasses.asdict`` recursively converts nested dataclasses to
        plain dicts.  Enum members (e.g. ``Containment.HALTED``) are copied
        as-is; because ``Containment`` inherits from ``str``, ``json.dumps``
        serialises them as their string value.
        """
        record = dataclasses.asdict(case)
        record["sponsor"] = sponsor
        return record

    @staticmethod
    def _deserialise(record: dict[str, Any]) -> tuple[RegressionCase, str]:
        """Reconstruct a ``RegressionCase`` and its sponsor from a stored dict.

        Raises ``CorpusIntegrityError`` (never a raw ``ValueError``/
        ``KeyError``) on any deserialization failure — a corrupted corpus must
        surface as the integrity error callers catch.

        The two correctness-critical steps (see module docstring):
        - ``ContainmentCase.expect`` is reconstructed via ``Containment(raw)``
          so it is an enum *member*, not a plain string.
        - ``SafetyProperty`` and ``ContainmentCase`` are built as dataclass
          *instances*, not left as plain dicts.
        """
        try:
            prop_d = record["property"]
            cc_d = record["containment_case"]

            prop = SafetyProperty(
                dim=prop_d["dim"],
                step=prop_d["step"],
                assertion=prop_d["assertion"],
                value=prop_d["value"],
                description=prop_d["description"],
            )

            # Containment(raw_str) raises ValueError if the string is not a
            # valid enum value; we catch that below and re-raise as
            # CorpusIntegrityError.
            cc = ContainmentCase(
                name=cc_d["name"],
                kind=cc_d["kind"],
                target=cc_d["target"],
                payload=cc_d["payload"],  # Any — preserved as-is (dict or list)
                expect=Containment(cc_d["expect"]),
            )

            case = RegressionCase(
                case_id=record["case_id"],
                property=prop,
                containment_case=cc,
                baseline_fingerprint=record["baseline_fingerprint"],
                initial_hash=record["initial_hash"],
                baseline_mocks_hash=record["baseline_mocks_hash"],
                derived_at=record["derived_at"],
                predecessor_hash=record["predecessor_hash"],
                case_hash=record["case_hash"],
            )
            sponsor: str = record["sponsor"]
        except (ValueError, KeyError) as exc:
            raise CorpusIntegrityError(
                f"failed to deserialise corpus record: {exc}"
            ) from exc

        return case, sponsor

    # --- disk scan ----------------------------------------------------------

    def _sorted_case_files(self) -> list[Path]:
        """Return all committed ``case-NNNNN.json`` files sorted by filename.

        Filename sort order equals append order (zero-padded numeric index).
        Temp files (``.tmp-*``) and any other non-case files are excluded.
        """
        return sorted(
            (
                f
                for f in self._root.iterdir()
                if f.name.startswith("case-")
                and f.name.endswith(".json")
                and ".tmp-" not in f.name
            ),
            key=lambda f: f.name,
        )

    def _load_from_disk(self) -> tuple[list[RegressionCase], list[str]]:
        """Read and deserialise all case files from disk in filename order.

        Raises ``CorpusIntegrityError`` on any read or deserialization failure.
        Callers that need tamper detection (``verify()``) call this directly,
        bypassing the in-memory cache.
        """
        cases: list[RegressionCase] = []
        sponsors: list[str] = []
        for path in self._sorted_case_files():
            try:
                with open(path, encoding="utf-8") as fh:
                    record = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                raise CorpusIntegrityError(
                    f"cannot read corpus case file {path.name!r}: {exc}"
                ) from exc
            case, sponsor = self._deserialise(record)
            cases.append(case)
            sponsors.append(sponsor)
        return cases, sponsors

    def _ensure_loaded(self) -> None:
        """Populate the in-memory cache from disk on the first call."""
        if self._cases_cache is None:
            self._cases_cache, self._sponsors_cache = self._load_from_disk()

    # --- RegressionCorpus interface -----------------------------------------

    def cases(self) -> list[RegressionCase]:
        """Return all cases in append order (cached after the first disk read)."""
        self._ensure_loaded()
        return list(self._cases_cache)  # type: ignore[arg-type]

    def sponsors(self) -> list[str]:
        """Return the sponsor for each case, in append order."""
        self._ensure_loaded()
        return list(self._sponsors_cache)  # type: ignore[arg-type]

    def root(self) -> str:
        """The hash of the newest case, or ``"genesis"`` if empty."""
        self._ensure_loaded()
        return (
            self._cases_cache[-1].case_hash  # type: ignore[index]
            if self._cases_cache
            else GENESIS
        )

    def append(self, case: RegressionCase, *, sponsor: str) -> None:
        """Append *case* to the corpus with *sponsor* attribution.

        Validates the sponsor, predecessor hash, and case hash before writing.
        Atomically writes the case file; updates the in-memory cache only after
        a successful write so a crash mid-write never corrupts the cache.
        """
        if not sponsor or not sponsor.strip():
            raise RatchetError(
                "corpus appends require a named human sponsor (got a blank string)"
            )
        self._ensure_loaded()
        current_root = (
            self._cases_cache[-1].case_hash  # type: ignore[index]
            if self._cases_cache
            else GENESIS
        )
        if case.predecessor_hash != current_root:
            raise RatchetError(
                f"case {case.case_id!r} declares predecessor "
                f"{case.predecessor_hash} but the chain tail is {current_root}"
            )
        if _recompute_hash(case) != case.case_hash:
            raise RatchetError(
                f"case {case.case_id!r} carries a hash that does not match its "
                f"content (recomputed hash differs)"
            )
        # Write to disk first; update cache only after a successful write.
        index = len(self._cases_cache) + 1  # type: ignore[arg-type]
        path = self._root / self._case_filename(index)
        self._atomic_write(path, self._serialise(case, sponsor.strip()))
        self._cases_cache.append(case)  # type: ignore[union-attr]
        self._sponsors_cache.append(sponsor.strip())  # type: ignore[union-attr]

    def verify(self) -> None:
        """Re-walk the on-disk chain; raise ``CorpusIntegrityError`` on any mismatch.

        **Always reads from disk** — never uses the in-memory cache — so that
        on-disk tampering is detected regardless of what the cache holds.
        Deserialization errors (e.g. a corrupted ``expect`` string that is not a
        valid ``Containment`` member, or a missing field) are caught and re-raised
        as ``CorpusIntegrityError`` with a legible message.
        """
        # _load_from_disk already wraps ValueError/KeyError as CorpusIntegrityError.
        cases, _ = self._load_from_disk()
        tail = GENESIS
        for case in cases:
            if case.predecessor_hash != tail:
                raise CorpusIntegrityError(
                    f"chain break at case {case.case_id!r}: predecessor "
                    f"{case.predecessor_hash} != expected {tail}"
                )
            if _recompute_hash(case) != case.case_hash:
                raise CorpusIntegrityError(
                    f"content tamper at case {case.case_id!r}: stored hash does "
                    f"not match recomputed hash"
                )
            tail = case.case_hash
