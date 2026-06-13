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


def _validate_sponsor(sponsor: str) -> None:
    """Fail closed on a blank or whitespace-only sponsor name."""
    if not sponsor or not sponsor.strip():
        raise RatchetError(
            "corpus appends require a named human sponsor (got a blank string)"
        )


def _assert_append_link(case: RegressionCase, current_root: str) -> None:
    """Guard the predecessor-hash and content-hash invariants for an append."""
    if case.predecessor_hash != current_root:
        raise RatchetError(
            f"case {case.case_id!r} declares predecessor {case.predecessor_hash} "
            f"but the chain tail is {current_root}"
        )
    if _recompute_hash(case) != case.case_hash:
        raise RatchetError(
            f"case {case.case_id!r} carries a hash that does not match its "
            f"content (recomputed hash differs)"
        )


def _walk_chain(cases: list[RegressionCase]) -> None:
    """Re-walk the hash chain; raise CorpusIntegrityError on any mismatch."""
    tail = GENESIS
    for case in cases:
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


def _first_required_field(model: type) -> str:
    for name, field in model.model_fields.items():
        if field.is_required():
            return name
    raise RatchetError(
        f"output model {model.__name__!r} has no required field; a schema "
        f"provocation cannot be derived for it"
    )


def derive_cases(
    config,
    pipeline,
    result,
    *,
    initial,
    baseline_mocks: Mapping,
    derived_at: str,
    predecessor: str | None,
):
    """Mechanically derive regression cases from an observed COMPLETED baseline.

    Cases are derived only for one-shot model steps (model set, no tools) whose
    pipeline prefix (every step at a smaller index — scheduler order) contains no
    model+tools step, so every replay can traverse the prefix using the pinned
    serializable mock subset. Derivation is the only authoring path: the
    provocation payloads come from the run's own validated outputs, never from a
    hand-written case.
    """
    from drawbore.escalation import HasConfidence
    from drawbore.pipeline.graph import JoinNode
    from drawbore.testing import Containment

    if result.status != "completed":
        raise RatchetError(
            f"baseline run must be completed to derive cases (got "
            f"status={result.status!r}) — a completed run proves the benign mock "
            f"bundle reaches every executed step"
        )
    mocks_hash = mocks_fingerprint(baseline_mocks)  # also validates the bundle keys
    initial_hash = canonical_fingerprint(initial.model_dump(mode="json"))
    baseline_fp = canonical_fingerprint(config.model_dump(mode="json"))

    executed_ok = {
        s.agent
        for s in (result.audit_trace.step_records if result.audit_trace else [])
        if s.status == "ok"
    }

    cases = []
    tail = predecessor if predecessor is not None else GENESIS
    counter = 0
    saw_loop = False

    def _seal(prop: SafetyProperty, cc: ContainmentCase):
        nonlocal tail, counter
        counter += 1
        case = seal_case(
            case_id=f"c-{counter:03d}",
            property=prop,
            containment_case=cc,
            baseline_fingerprint=baseline_fp,
            initial_hash=initial_hash,
            baseline_mocks_hash=mocks_hash,
            derived_at=derived_at,
            predecessor_hash=tail,
        )
        tail = case.case_hash
        cases.append(case)

    for node in pipeline.steps:
        if isinstance(node, JoinNode):
            continue
        spec = node.agent.spec
        is_loop = spec.model is not None and bool(spec.tools)
        is_one_shot = spec.model is not None and not spec.tools
        if is_loop:
            saw_loop = True
            continue
        if not is_one_shot or saw_loop or spec.name not in executed_ok:
            continue
        observed = result.outputs[spec.name].model_dump(mode="json")

        bad = dict(observed)
        bad.pop(_first_required_field(spec.output), None)
        _seal(
            SafetyProperty(
                dim="schema_violations", step=spec.name, assertion="neq", value="0",
                description=(
                    f"an output from '{spec.name}' violating its declared schema "
                    f"must be rejected"
                ),
            ),
            ContainmentCase(
                name=f"schema_violation:{spec.name}", kind="schema_violation",
                target=spec.name, payload=bad, expect=Containment.SCHEMA_REJECT,
            ),
        )

        if issubclass(spec.output, HasConfidence):
            low = dict(observed)
            low["confidence"] = 0.0
            expect = (
                Containment.ESCALATED if pipeline.on_failure is not None
                else Containment.HALTED
            )
            _seal(
                SafetyProperty(
                    dim="run_status", step=spec.name, assertion="neq",
                    value="completed",
                    description=(
                        f"a below-threshold confidence from '{spec.name}' must "
                        f"not complete the run (pinned refusal: {expect.value})"
                    ),
                ),
                ContainmentCase(
                    name=f"low_confidence:{spec.name}", kind="low_confidence",
                    target=spec.name, payload=low, expect=expect,
                ),
            )

    if not cases:
        raise RatchetError(
            "no derivable cases: the pipeline has no one-shot model step ahead of "
            "its first model+tools step, so a ratchet over it would be vacuous"
        )
    return cases


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
        _validate_sponsor(sponsor)
        current_root = self.root()
        _assert_append_link(case, current_root)
        self._cases.append(case)
        self._sponsors.append(sponsor.strip())

    def verify(self) -> None:
        _walk_chain(self._cases)
