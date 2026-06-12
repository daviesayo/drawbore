"""The admission gate: the only sanctioned path from a candidate manifest to a
blessed ``Pipeline``.

Layer order (each failure is a legible non-admit verdict, corpus root unchanged):
corpus integrity (chain verify + pinned replay-input hash checks) -> fail-closed
resolver -> authority monotonicity (widening returns the diff for HUMAN review,
never auto-admits) -> regression-corpus replay through the real safety layer.
On admission, corpus growth is best-effort and deduplicated — never a gate.
There is no skip flag and no partial-admission mode.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from pydantic import BaseModel, ValidationError

from drawbore._canon import canonical_fingerprint
from drawbore.agent import Agent
from drawbore.config import AgentCatalog, from_config
from drawbore.config.errors import ConfigResolutionError
from drawbore.config.models import PipelineConfig
from drawbore.testing import run_containment
from drawbore.testing.errors import TestingError

from .capability import authority_delta
from .corpus import RegressionCorpus, derive_cases, mocks_fingerprint, seal_case
from .errors import CorpusIntegrityError, RatchetError
from .sink import RatchetSink
from .verdict import RatchetVerdict


def _parse(candidate_manifest: "str | bytes | Mapping[str, Any]") -> PipelineConfig:
    if isinstance(candidate_manifest, (str, bytes, bytearray)):
        raw = json.loads(candidate_manifest)
    else:
        raw = dict(candidate_manifest)
    return PipelineConfig.model_validate(raw)


async def admit(
    candidate_manifest: "str | bytes | Mapping[str, Any]",
    *,
    agents: "AgentCatalog | Mapping[str, Agent]",
    corpus: RegressionCorpus,
    sponsor: str,
    baseline_config: PipelineConfig,
    initial_input: BaseModel,
    baseline_mocks: Mapping,
    derived_at: str,
    registry: Any = None,
    sink: "RatchetSink | None" = None,
) -> RatchetVerdict:
    """Decide one candidate manifest. Returns a frozen verdict; the blessed
    ``Pipeline`` rides on it iff admitted. Adoption is reference replacement by
    the caller — there is no service to hot-swap."""
    root_before = corpus.root()

    def _done(verdict: RatchetVerdict) -> RatchetVerdict:
        if sink is not None:
            sink.write(verdict)
        return verdict

    def _reject(layer, *, fingerprint="", diff=None, case=None, reason=None):
        return _done(RatchetVerdict(
            admitted=False, pipeline=None,
            manifest_fingerprint=fingerprint,
            rejection_layer=layer, authority_diff=diff,
            failed_case_id=case.case_id if case is not None else None,
            failed_property_description=(
                case.property.description if case is not None else None
            ),
            corpus_root_before=root_before, corpus_root_after=None,
            reason=reason,
        ))

    # Layer 0: corpus integrity — the chain itself, then the pinned replay inputs.
    try:
        corpus.verify()
    except CorpusIntegrityError as exc:
        return _reject("corpus_integrity", reason=str(exc))
    try:
        mocks_hash = mocks_fingerprint(baseline_mocks)
    except RatchetError as exc:
        return _reject("corpus_integrity", reason=str(exc))
    initial_hash = canonical_fingerprint(initial_input.model_dump(mode="json"))
    for case in corpus.cases():
        if case.initial_hash != initial_hash or case.baseline_mocks_hash != mocks_hash:
            return _reject(
                "corpus_integrity",
                reason=(
                    f"replay inputs do not match what case {case.case_id!r} pinned "
                    f"at derivation (initial and mock bundle are frozen with the chain)"
                ),
            )

    # Layer 1: parse once + fail-closed resolution.
    try:
        candidate_config = _parse(candidate_manifest)
    except (ValidationError, TypeError, ValueError) as exc:
        return _reject("resolver", reason=f"candidate manifest did not parse: {exc}")
    fingerprint = canonical_fingerprint(candidate_config.model_dump(mode="json"))
    try:
        candidate_pipeline = from_config(candidate_config, agents=agents, registry=registry)
    except ConfigResolutionError as exc:
        return _reject("resolver", fingerprint=fingerprint, reason=str(exc))

    # Layer 2: authority monotonicity — widening is for a human, never auto-admit.
    diff = authority_delta(baseline_config, candidate_config)
    if not diff.ok:
        return _reject("authority", fingerprint=fingerprint, diff=diff)

    # Layer 3: corpus replay through the real safety layer.
    for case in corpus.cases():
        observed = await run_containment(
            candidate_pipeline, case.containment_case,
            initial=initial_input, baseline=dict(baseline_mocks),
        )
        if observed is not case.containment_case.expect:
            return _reject("corpus", fingerprint=fingerprint, case=case)

    # Admission. Growth is best-effort, deduplicated, and never a gate.
    root_after: str | None = None
    try:
        async with candidate_pipeline.test_mode(**{
            k: dict(v) for k, v in baseline_mocks.items()
        }) as tp:
            growth_run = await tp.run(initial_input)
        if growth_run.status == "completed":
            existing = {
                (c.containment_case.target, c.containment_case.kind,
                 canonical_fingerprint(c.containment_case.payload))
                for c in corpus.cases()
            }
            fresh = derive_cases(
                candidate_config, candidate_pipeline, growth_run,
                initial=initial_input, baseline_mocks=baseline_mocks,
                derived_at=derived_at, predecessor=corpus.root(),
            )
            appended = False
            tail = corpus.root()
            for case in fresh:
                key = (case.containment_case.target, case.containment_case.kind,
                       canonical_fingerprint(case.containment_case.payload))
                if key in existing:
                    continue
                # re-seal onto the live tail (dedup may have skipped predecessors)
                resealed = seal_case(
                    case_id=f"c-{len(corpus.cases()) + 1:03d}",
                    property=case.property,
                    containment_case=case.containment_case,
                    baseline_fingerprint=case.baseline_fingerprint,
                    initial_hash=case.initial_hash,
                    baseline_mocks_hash=case.baseline_mocks_hash,
                    derived_at=case.derived_at,
                    predecessor_hash=tail,
                )
                corpus.append(resealed, sponsor=sponsor)
                tail = resealed.case_hash
                appended = True
            if appended:
                root_after = corpus.root()
    except (TestingError, RatchetError):
        # candidate does not complete under the pinned inputs, or derivation found
        # nothing — growth is opportunistic; the admission stands.
        root_after = None

    return _done(RatchetVerdict(
        admitted=True, pipeline=candidate_pipeline,
        manifest_fingerprint=fingerprint,
        rejection_layer=None, authority_diff=None,
        failed_case_id=None, failed_property_description=None,
        corpus_root_before=root_before, corpus_root_after=root_after,
    ))
