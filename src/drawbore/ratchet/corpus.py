"""The policy-regression corpus: hash-chained, append-only, mechanically derived.

A case freezes one observed safety property of a baseline pipeline as a
provocation (a gauntlet containment case) plus a legible assertion. The chain is
a predecessor-hash chain over canonical-JSON SHA-256; the root is the newest
case hash (or the literal sentinel "genesis" when empty). Appends record a named human sponsor — attribution, not cryptographic custody; the hash chain, not the sponsor record, is the tamper-evidence.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Mapping

from drawbore._canon import canonical_fingerprint
from drawbore.testing import ContainmentCase

from .errors import CorpusIntegrityError, RatchetError

GENESIS = "genesis"

_ALLOWED_MOCK_KEYS = ("mock_model_responses", "mock_tools")


@dataclass(frozen=True)
class SafetyProperty:
    """A legible, audit-checkable assertion — data, never a lambda."""

    dim: Literal["run_status", "step_status", "escalated", "schema_violations", "halt_code"]
    step: str | None
    assertion: Literal["eq", "neq", "in"]
    value: str
    description: str


@dataclass(frozen=True)
class RegressionCase:
    """One frozen safety property with its provocation and pinned replay inputs."""

    case_id: str
    property: SafetyProperty
    containment_case: ContainmentCase
    baseline_fingerprint: str
    initial_hash: str
    baseline_mocks_hash: str
    derived_at: str
    predecessor_hash: str
    case_hash: str


def _case_digest_payload(
    *,
    case_id: str,
    property: SafetyProperty,
    containment_case: ContainmentCase,
    baseline_fingerprint: str,
    initial_hash: str,
    baseline_mocks_hash: str,
    derived_at: str,
    predecessor_hash: str,
) -> dict:
    return {
        "case_id": case_id,
        "property": dataclasses.asdict(property),
        "containment_case": dataclasses.asdict(containment_case),
        "baseline_fingerprint": baseline_fingerprint,
        "initial_hash": initial_hash,
        "baseline_mocks_hash": baseline_mocks_hash,
        "derived_at": derived_at,
        "predecessor_hash": predecessor_hash,
    }


def seal_case(
    *,
    case_id: str,
    property: SafetyProperty,
    containment_case: ContainmentCase,
    baseline_fingerprint: str,
    initial_hash: str,
    baseline_mocks_hash: str,
    derived_at: str,
    predecessor_hash: str,
) -> RegressionCase:
    """Construct a case with its hash computed over every other field (pinned
    serialization: plain dict via ``dataclasses.asdict``, canonical-JSON SHA-256)."""
    digest = canonical_fingerprint(
        _case_digest_payload(
            case_id=case_id,
            property=property,
            containment_case=containment_case,
            baseline_fingerprint=baseline_fingerprint,
            initial_hash=initial_hash,
            baseline_mocks_hash=baseline_mocks_hash,
            derived_at=derived_at,
            predecessor_hash=predecessor_hash,
        )
    )
    return RegressionCase(
        case_id=case_id,
        property=property,
        containment_case=containment_case,
        baseline_fingerprint=baseline_fingerprint,
        initial_hash=initial_hash,
        baseline_mocks_hash=baseline_mocks_hash,
        derived_at=derived_at,
        predecessor_hash=predecessor_hash,
        case_hash=digest,
    )


def _recompute_hash(case: RegressionCase) -> str:
    return canonical_fingerprint(
        _case_digest_payload(
            case_id=case.case_id,
            property=case.property,
            containment_case=case.containment_case,
            baseline_fingerprint=case.baseline_fingerprint,
            initial_hash=case.initial_hash,
            baseline_mocks_hash=case.baseline_mocks_hash,
            derived_at=case.derived_at,
            predecessor_hash=case.predecessor_hash,
        )
    )


def mocks_fingerprint(baseline_mocks: Mapping) -> str:
    """The pinned fingerprint of a benign mock bundle: canonical-JSON SHA-256
    over exactly the two serializable keys, both always present. A bundle that
    names any other key (notably loop scripts, which are code-like objects and
    not canonically serializable) is rejected — fail closed and legible."""
    extra = set(baseline_mocks) - set(_ALLOWED_MOCK_KEYS)
    if extra:
        raise RatchetError(
            f"baseline_mocks may only contain {list(_ALLOWED_MOCK_KEYS)}; "
            f"got unsupported key(s) {sorted(extra)} (mock_loop_scripts and other "
            f"non-serializable mocks cannot be pinned into the corpus)"
        )
    normalized = {key: dict(baseline_mocks.get(key, {})) for key in _ALLOWED_MOCK_KEYS}
    return canonical_fingerprint(normalized)


class RegressionCorpus(ABC):
    """Append-only store for the case chain."""

    @abstractmethod
    def cases(self) -> list[RegressionCase]: ...

    @abstractmethod
    def append(self, case: RegressionCase, *, sponsor: str) -> None: ...

    @abstractmethod
    def sponsors(self) -> list[str]:
        """The recorded sponsor for each appended case, in append order."""

    @abstractmethod
    def root(self) -> str: ...

    @abstractmethod
    def verify(self) -> None:
        """Re-walk the chain; raise ``CorpusIntegrityError`` on any mismatch."""


class InMemoryRegressionCorpus(RegressionCorpus):
    """In-process default. Appends are sponsor-attributed and chain-checked."""

    def __init__(self) -> None:
        self._cases: list[RegressionCase] = []
        self._sponsors: list[str] = []

    def cases(self) -> list[RegressionCase]:
        return list(self._cases)

    def sponsors(self) -> list[str]:
        return list(self._sponsors)

    def root(self) -> str:
        return self._cases[-1].case_hash if self._cases else GENESIS

    def append(self, case: RegressionCase, *, sponsor: str) -> None:
        if not sponsor or not sponsor.strip():
            raise RatchetError(
                "corpus appends require a named human sponsor (got a blank string)"
            )
        if case.predecessor_hash != self.root():
            raise RatchetError(
                f"case {case.case_id!r} declares predecessor {case.predecessor_hash} "
                f"but the chain tail is {self.root()}"
            )
        if _recompute_hash(case) != case.case_hash:
            raise RatchetError(
                f"case {case.case_id!r} carries a hash that does not match its "
                f"content (recomputed hash differs)"
            )
        self._cases.append(case)
        self._sponsors.append(sponsor.strip())

    def verify(self) -> None:
        tail = GENESIS
        for case in self._cases:
            if case.predecessor_hash != tail:
                raise CorpusIntegrityError(
                    f"chain break at case {case.case_id!r}: predecessor "
                    f"{case.predecessor_hash} != expected {tail}"
                )
            if _recompute_hash(case) != case.case_hash:
                raise CorpusIntegrityError(
                    f"content tamper at case {case.case_id!r}: stored hash does not "
                    f"match recomputed hash"
                )
            tail = case.case_hash
