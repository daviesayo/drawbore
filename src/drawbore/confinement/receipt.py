"""Confinement receipt: a deterministic, offline-verifiable run attestation.

A ``ConfinementReceipt`` binds a run's *declared* confinement (its effective
authority footprint) to its *observed* enforcement (the proxy's tool-call log +
per-step taint scope), and ``verify`` independently re-checks the confinement
invariants and the artifact's integrity. The artifact is FINGERPRINTED with a
deterministic, keyless SHA-256 over a fixed positional payload, NOT
cryptographically signed.

Tamper-evidence is layered, with honest scope for each layer:

- ``verify(receipt)`` re-derives the fingerprint and re-evaluates the verdict.
  This detects edits that did NOT recompute the keyless fingerprint, and (via an
  explicit consistency check) also detects a ``footprint_fingerprint`` field that
  is inconsistent with ``declared_facts``. It does NOT detect a fully
  self-consistent forgery in which the adversary rewrites ``declared_facts`` AND
  recomputes ``footprint_fingerprint`` AND recomputes ``receipt_fingerprint`` —
  all with the same public SHA-256 algorithm.
- ``verify(receipt, proxy_log)`` additionally binds the receipt to a specific
  execution log, catching a substituted log.
- ``verify(receipt, expected_footprint_fingerprint=...)`` binds the declared
  footprint to the manifest fingerprint the auditor holds (e.g. from
  ``effective_authority(config).fingerprint()``), catching an inflated
  ``declared_facts`` even when ALL keyless fingerprints are self-consistent.

Non-repudiable signing (Ed25519) is a separate, out-of-core concern.

A run is CONFINED iff, over every *executed* tool call (``result`` in
``{"ok", "replay"}``):

1. **Footprint containment** — the call maps to a declared capability: a fact
   with ``(subject == agent_of_step, ref == tool_ref)`` exists. Match is on
   ``(subject, ref)`` ONLY (neither the resolved tool kind nor the scope), which
   is the sound criterion: a legitimately-callable tool always has a matching
   declared fact, an undeclared tool has none.
2. **Taint no-exfil** — no executed ``exfil_capable`` call had
   ``taint_scope == "untrusted"``.

A denied/errored call is never a breach (a fired control is evidence, not a
violation). Insufficient evidence to *prove* containment (a missing footprint, a
call that cannot be attributed to an agent, or an integrity mismatch) is
UNVERIFIABLE, never silently CONFINED — fail closed. Verdict precedence:
``unverifiable > breached > confined``.

This module imports only ``_canon``, the config layer (the footprint fingerprint
helper + the evidence ref constant), Pydantic, and stdlib. ``verify`` and the
receipt types touch no pipeline/proxy/orchestration code; only the pipeline's
mint call reaches inward.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from drawbore._canon import canonical_fingerprint

# Reuse the evidence-retrieval ref rather than hardcoding a literal. The config
# authority layer mirrors it (keeping its own import boundary); we reuse that
# mirror so the confinement module stays inside its allowed import set.
from drawbore.config.authority import EVIDENCE_RETRIEVE_REF as EVIDENCE_REF
from drawbore.config.fingerprint import footprint_fingerprint

__all__ = [
    "ObservedCall",
    "ConfinementVerdict",
    "ConfinementReceipt",
    "verify",
    "mint_receipt",
    "EVIDENCE_REF",
]

# Results that count as an EXECUTED tool call (a footprint/taint candidate).
_EXECUTED = ("ok", "replay")

VerdictStatus = Literal["confined", "breached", "unverifiable"]
RunStatus = Literal["completed", "halted", "escalated"]


class ObservedCall(BaseModel):
    """One proxy-log tool call, as the receipt records it. Frozen."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int | None
    agent: str | None
    tool_ref: str
    operation: str
    match_kind: str | None  # "tool" | "evidence" | None (legibility only)
    result: str  # ok | replay | error | denied:token|breaker|scope|taint
    taint_scope: str  # "trusted" | "untrusted"
    exfil_capable: bool
    input_hash: str | None
    output_hash: str | None


class ConfinementVerdict(BaseModel):
    """The confinement outcome over a set of observed calls. Frozen.

    ``confined`` MUST equal ``status == "confined"``; a validator enforces it so
    the two cannot drift into an internally inconsistent (but otherwise
    fingerprint-valid) receipt.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: VerdictStatus
    confined: bool
    breaches: tuple[str, ...]
    unverifiable: tuple[str, ...]

    @model_validator(mode="after")
    def _confined_matches_status(self) -> "ConfinementVerdict":
        if self.confined != (self.status == "confined"):
            raise ValueError(
                "ConfinementVerdict.confined must equal (status == 'confined')"
            )
        return self


class ConfinementReceipt(BaseModel):
    """A self-contained, offline-verifiable confinement attestation. Frozen.

    ``run_id`` defaults (at the pipeline) to a non-unique ``"<name>-run"``; pass
    an explicit unique ``run_id`` for per-run receipt identity. NOTE the two
    ``status`` fields: this one is the RUN status, while ``verdict.status`` is the
    CONFINEMENT verdict.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    status: RunStatus
    footprint_fingerprint: str
    declared_facts: tuple[tuple[str, str, str, str], ...]
    step_agents: tuple[str | None, ...]
    observed_calls: tuple[ObservedCall, ...]
    verdict: ConfinementVerdict
    receipt_fingerprint: str

    def legible(self) -> str:
        """A regulator-readable, single-block account of the receipt."""
        lines = [
            f"Confinement receipt for run {self.run_id} (run status: {self.status})."
        ]
        lines.append(
            f"Declared footprint {self.footprint_fingerprint} — "
            f"{len(self.declared_facts)} capability fact(s):"
        )
        for subject, kind, ref, scope in sorted(self.declared_facts):
            lines.append(f"  {subject} may use {kind} {ref} (scope '{scope}')")
        lines.append(f"Observed {len(self.observed_calls)} tool call(s):")
        for c in self.observed_calls:
            agent = c.agent if c.agent is not None else "<unattributed>"
            lines.append(
                f"  step {c.step} {agent} -> {c.tool_ref} ({c.operation}) "
                f"[{c.match_kind}] result={c.result} taint={c.taint_scope} "
                f"exfil_capable={c.exfil_capable}"
            )
        v = self.verdict
        lines.append(f"Verdict: {v.status.upper()}.")
        for b in v.breaches:
            lines.append(f"  BREACH: {b}")
        for u in v.unverifiable:
            lines.append(f"  UNVERIFIABLE: {u}")
        lines.append(f"Receipt fingerprint {self.receipt_fingerprint}.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared helpers (used by both mint_receipt and verify)
# ---------------------------------------------------------------------------


def _observed_from_log_entry(
    entry: dict, step_agents: tuple[str | None, ...]
) -> ObservedCall:
    """Derive one ``ObservedCall`` from a proxy-log entry.

    The SINGLE transform used by both ``mint_receipt`` and ``verify``'s
    log-binding step, so a re-derivation from the live log is identical to what
    the minter embedded. ``step`` indexes ``step_agents`` to recover the agent
    (``None`` if the step is missing/out of range — an unattributable call).
    ``match_kind`` is normalised to the fact vocabulary: ``"evidence"`` for the
    evidence ref, ``"tool"`` for any other resolved tool, ``None`` when the tool
    was never resolved (an undeclared-tool denial).
    """
    step = entry.get("step")
    agent: str | None = None
    if isinstance(step, int) and 0 <= step < len(step_agents):
        agent = step_agents[step]
    tool_ref = entry["tool"]
    kind = entry.get("kind")
    if tool_ref == EVIDENCE_REF:
        match_kind: str | None = "evidence"
    elif kind is not None:
        match_kind = "tool"
    else:
        match_kind = None
    return ObservedCall(
        step=step,
        agent=agent,
        tool_ref=tool_ref,
        operation=entry["operation"],
        match_kind=match_kind,
        result=entry["result"],
        taint_scope=entry["scope"],
        exfil_capable=bool(entry.get("exfil_capable", False)),
        input_hash=entry.get("input_hash"),
        output_hash=entry.get("output_hash"),
    )


def _evaluate(
    observed_calls: tuple[ObservedCall, ...],
    declared_facts: tuple[tuple[str, str, str, str], ...],
) -> ConfinementVerdict:
    """Evaluate the two confinement invariants over observed calls.

    Match footprint containment on ``(subject, ref)`` only. Denied/errored calls
    are not candidates. Precedence: ``unverifiable > breached > confined``.
    """
    declared = {(f[0], f[2]) for f in declared_facts}
    breaches: list[str] = []
    unverifiable: list[str] = []
    for c in observed_calls:
        if c.result not in _EXECUTED:
            continue  # a denial/error is evidence a control fired, never a breach
        if c.agent is None:
            unverifiable.append(
                f"step {c.step} call to {c.tool_ref} could not be attributed "
                f"to an agent"
            )
            continue
        if (c.agent, c.tool_ref) not in declared:
            breaches.append(
                f"step {c.step} agent {c.agent} called undeclared tool {c.tool_ref}"
            )
        if c.exfil_capable and c.taint_scope == "untrusted":
            breaches.append(
                f"exfil-capable {c.tool_ref} executed under untrusted scope in "
                f"step {c.step}"
            )
    if unverifiable:
        return ConfinementVerdict(
            status="unverifiable",
            confined=False,
            breaches=tuple(breaches),
            unverifiable=tuple(unverifiable),
        )
    if breaches:
        return ConfinementVerdict(
            status="breached", confined=False, breaches=tuple(breaches), unverifiable=()
        )
    return ConfinementVerdict(
        status="confined", confined=True, breaches=(), unverifiable=()
    )


def _fingerprint(
    run_id: str,
    status: str,
    footprint_fp: str,
    declared_facts: tuple[tuple[str, str, str, str], ...],
    step_agents: tuple[str | None, ...],
    observed_calls: tuple[ObservedCall, ...],
    verdict: ConfinementVerdict,
) -> str:
    """Return the canonical fingerprint over a FIXED, POSITIONAL, JSON-safe payload.

    Omits ``receipt_fingerprint`` itself. ``declared_facts`` elements are
    4-tuples (``list(t)``); ``step_agents`` elements are scalars (``list(...)``,
    NOT per-element splitting). Both the minter and ``verify`` build this exact
    list (NOT ``model_dump()`` of the whole receipt, whose dict-key sorting
    differs).
    """
    return canonical_fingerprint(
        [
            run_id,
            status,
            footprint_fp,
            sorted([list(t) for t in declared_facts]),
            list(step_agents),
            [c.model_dump(mode="json") for c in observed_calls],
            verdict.model_dump(mode="json"),
        ]
    )


def _unverifiable(reason: str) -> ConfinementVerdict:
    return ConfinementVerdict(
        status="unverifiable", confined=False, breaches=(), unverifiable=(reason,)
    )


# ---------------------------------------------------------------------------
# Public verifier
# ---------------------------------------------------------------------------


def verify(
    receipt: ConfinementReceipt,
    proxy_log: list[dict] | None = None,
    *,
    expected_footprint_fingerprint: str | None = None,
) -> ConfinementVerdict:
    """Offline, deterministic re-verification of a confinement receipt.

    ``verify(receipt)`` (no log) re-derives the receipt fingerprint and
    re-evaluates the confinement verdict. It detects edits that did NOT
    recompute the keyless fingerprint, and also detects a
    ``footprint_fingerprint`` field that is inconsistent with ``declared_facts``.
    It does NOT detect a fully self-consistent keyless forgery — see
    ``expected_footprint_fingerprint`` for that protection.

    ``verify(receipt, proxy_log)`` additionally binds the receipt to an actual
    execution by re-deriving the observed calls from the live log and asserting
    they equal the embedded calls. Each log entry's ``run_id`` is also checked
    against ``receipt.run_id``, so a different run's log with identical calls is
    rejected.

    ``verify(receipt, expected_footprint_fingerprint=fp)`` additionally asserts
    that ``receipt.footprint_fingerprint`` equals ``fp`` — the footprint
    fingerprint the auditor independently holds (e.g. from
    ``effective_authority(config).fingerprint()``). This closes the self-consistent
    keyless forgery gap: even if an adversary inflates ``declared_facts`` and
    recomputes all keyless fingerprints consistently, the inflated footprint will
    not match the manifest fingerprint the auditor holds independently.

    Fail closed: any inability to re-derive returns an UNVERIFIABLE (not-confined)
    verdict; no exception escapes.
    """
    try:
        # Step 1 — overall integrity: re-derive receipt_fingerprint over the
        # fixed positional payload (including footprint_fingerprint as a field).
        # Detects any edit that did not also recompute the keyless fingerprint.
        expected_fp = _fingerprint(
            receipt.run_id,
            receipt.status,
            receipt.footprint_fingerprint,
            receipt.declared_facts,
            receipt.step_agents,
            receipt.observed_calls,
            receipt.verdict,
        )
        if expected_fp != receipt.receipt_fingerprint:
            return _unverifiable("receipt fingerprint mismatch — tampered")

        # Step 2 — footprint consistency: the footprint_fingerprint field must be
        # the canonical fingerprint of the declared_facts it claims to summarise.
        # Catches a forger who changes declared_facts but leaves the original
        # footprint_fingerprint (or vice versa).
        recomputed_footprint_fp = footprint_fingerprint(receipt.declared_facts)
        if recomputed_footprint_fp != receipt.footprint_fingerprint:
            return _unverifiable(
                "footprint fingerprint does not match declared facts"
            )

        # Step 3 — expected footprint binding (optional): assert the declared
        # footprint equals the manifest fingerprint the auditor independently holds.
        # Catches a fully self-consistent keyless forgery that inflates declared_facts
        # and recomputes all fingerprints — which Step 2 alone cannot detect.
        if expected_footprint_fingerprint is not None:
            if expected_footprint_fingerprint != receipt.footprint_fingerprint:
                return _unverifiable(
                    "declared footprint does not match the expected (manifest) footprint"
                )

        # Step 4 — verdict re-derivation: re-evaluate the two confinement
        # invariants over the embedded observed calls and declared facts.
        rederived = _evaluate(receipt.observed_calls, receipt.declared_facts)
        if rederived != receipt.verdict:
            return _unverifiable("verdict does not re-derive")

        # Step 5 — log binding (optional): re-derive observed calls from the live
        # proxy log and assert they match the embedded calls. Also checks that every
        # log entry belongs to this run (run_id match), so a different run's log
        # with byte-identical calls cannot pass as binding evidence.
        if proxy_log is not None:
            for entry in proxy_log:
                if entry.get("run_id") != receipt.run_id:
                    return _unverifiable(
                        "supplied proxy log is for a different run"
                    )
            recalls = tuple(
                _observed_from_log_entry(entry, receipt.step_agents)
                for entry in proxy_log
            )
            if recalls != receipt.observed_calls:
                return _unverifiable("receipt does not match the supplied proxy log")

        return receipt.verdict
    except Exception:  # fail closed — never let a verifier error masquerade as ok
        return _unverifiable("receipt could not be verified")


# ---------------------------------------------------------------------------
# Minting (built from a proxy log; the pipeline calls this in run()'s finally)
# ---------------------------------------------------------------------------


def mint_receipt(
    *,
    run_id: str,
    run_status: RunStatus,
    proxy_log: list[dict],
    step_agents: tuple[str | None, ...],
    footprint: Any | None = None,
) -> ConfinementReceipt:
    """Mint a ``ConfinementReceipt`` from a proxy log + step→agent map.

    ``footprint`` is the declared ``CapabilityFootprint`` (``effective_authority``
    of the run's resolved config); when ``None`` (the footprint could not be
    derived) the receipt is UNVERIFIABLE — fail closed. Observed calls are built
    via the shared ``_observed_from_log_entry`` transform so a later offline
    ``verify(receipt, log)`` re-derives identically.

    This function is TOTAL: any internal failure (e.g. a malformed proxy-log
    entry that causes a ``KeyError`` in ``_observed_from_log_entry``) returns a
    deterministic UNVERIFIABLE receipt — it never raises and never returns
    ``None``. The failure receipt carries empty ``declared_facts`` and
    ``observed_calls``, a valid ``receipt_fingerprint``, and an unverifiable
    verdict whose reason names the exception class.
    """
    try:
        step_agents = tuple(step_agents)
        # An empty observed-call set is a legitimate "this run made no tool calls"
        # attestation. A run with zero calls is CONFINED if the footprint is present;
        # this is not a missing-evidence case.
        observed = tuple(
            _observed_from_log_entry(entry, step_agents) for entry in proxy_log
        )

        if footprint is None:
            declared_facts: tuple[tuple[str, str, str, str], ...] = ()
            verdict = _unverifiable("declared footprint unavailable")
            footprint_fp = footprint_fingerprint(())
        else:
            declared_facts = tuple(
                sorted((f.subject, f.kind, f.ref, f.scope) for f in footprint.facts)
            )
            verdict = _evaluate(observed, declared_facts)
            footprint_fp = footprint.fingerprint()

        receipt_fp = _fingerprint(
            run_id, run_status, footprint_fp, declared_facts, step_agents, observed, verdict
        )
        return ConfinementReceipt(
            run_id=run_id,
            status=run_status,
            footprint_fingerprint=footprint_fp,
            declared_facts=declared_facts,
            step_agents=step_agents,
            observed_calls=observed,
            verdict=verdict,
            receipt_fingerprint=receipt_fp,
        )
    except Exception as exc:  # noqa: BLE001  — fail closed; never let mint raise
        _empty_facts: tuple[tuple[str, str, str, str], ...] = ()
        _empty_calls: tuple[ObservedCall, ...] = ()
        _fail_verdict = _unverifiable(
            f"receipt minting failed: {type(exc).__name__}"
        )
        _empty_fp = footprint_fingerprint(_empty_facts)
        _receipt_fp = _fingerprint(
            run_id, run_status, _empty_fp, _empty_facts, (), _empty_calls, _fail_verdict
        )
        return ConfinementReceipt(
            run_id=run_id,
            status=run_status,
            footprint_fingerprint=_empty_fp,
            declared_facts=_empty_facts,
            step_agents=(),
            observed_calls=_empty_calls,
            verdict=_fail_verdict,
            receipt_fingerprint=_receipt_fp,
        )
