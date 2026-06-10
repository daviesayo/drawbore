"""Effective-authority footprint. Pure config layer.

Derives the statically reachable capability footprint from a PipelineConfig
manifest: the set of (subject, kind, ref, scope) facts the pipeline is *allowed*
to exercise. Tool + evidence-handle granular and guard-blind (operations collapse
to "*", branch guards ignored) — a sound over-approximation.
Imports only the config layer; no tools/runtime imports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .errors import AuthorityRegressionError
from .fingerprint import footprint_fingerprint
from .models import PipelineConfig, StepConfig

# The evidence retrieval builtin's ref. Mirrors evidence.retrieval.EVIDENCE_TOOL_REF,
# duplicated as a literal so the config layer keeps its import boundary (no evidence
# import). If that ref ever changes, this must change with it.
EVIDENCE_RETRIEVE_REF = "evidence://retrieve"


@dataclass(frozen=True)
class CapabilityFact:
    subject: str   # agent/step name holding the authority
    kind: str      # "tool" | "evidence"
    ref: str       # "mcp://identity/verify" | "evidence://retrieve"
    scope: str     # tool: "*"; evidence: "search" | "full"


@dataclass(frozen=True)
class CapabilityFootprint:
    facts: frozenset[CapabilityFact]

    def fingerprint(self) -> str:
        return footprint_fingerprint(
            (f.subject, f.kind, f.ref, f.scope) for f in self.facts
        )


def effective_authority(config: PipelineConfig) -> CapabilityFootprint:
    """The statically reachable capability footprint of a manifest (guard-blind)."""
    agents = {a.name: a for a in config.agents}
    facts: set[CapabilityFact] = set()
    for node in config.steps:
        if not isinstance(node, StepConfig):
            continue  # JoinConfig grants no tool/evidence authority
        agent = agents.get(node.agent)
        if agent is None:
            continue  # the resolver enforces existence; footprint stays a sound subset
        for tool_ref in agent.tools:
            facts.add(CapabilityFact(node.agent, "tool", tool_ref, "*"))
        ev = node.evidence
        if ev is not None and ev.enabled and agent.model is not None:
            # An enabled evidence policy on a model agent declares evidence-handling
            # authority (compression and/or retrieval). v1 emits both scope facts from
            # the policy flags — a sound over-approximation (a compress-only agent that
            # never calls evidence://retrieve still has the policy permitting it). The
            # `enabled and model is not None` gate mirrors pipeline/executor.py:90.
            if ev.allow_search_retrieval:
                facts.add(CapabilityFact(node.agent, "evidence", EVIDENCE_RETRIEVE_REF, "search"))
            if ev.allow_full_retrieval:
                facts.add(CapabilityFact(node.agent, "evidence", EVIDENCE_RETRIEVE_REF, "full"))
    return CapabilityFootprint(frozenset(facts))


# ---------------------------------------------------------------------------
# Diff + certificate (Task 3)
# ---------------------------------------------------------------------------

def _phrase(modal: str, f: CapabilityFact) -> str:
    if f.kind == "tool":
        return f"{f.subject} {modal} call tool {f.ref}"
    return f"{f.subject} {modal} read evidence {f.ref} at scope '{f.scope}'"


def _sort_key(f: CapabilityFact) -> tuple[str, str, str, str]:
    return (f.subject, f.kind, f.ref, f.scope)


@dataclass(frozen=True)
class AuthorityDiff:
    old_fingerprint: str
    new_fingerprint: str
    added: frozenset[CapabilityFact]
    removed: frozenset[CapabilityFact]

    @property
    def ok(self) -> bool:
        return not self.added

    def certificate(self) -> str:
        status = "PASSED" if self.ok else "FAILED"
        lines = [
            f"Authority regression check: {status} "
            f"— {len(self.added)} new capability(ies) granted."
        ]
        for f in sorted(self.added, key=_sort_key):
            lines.append("+ " + _phrase("may", f))
        for f in sorted(self.removed, key=_sort_key):
            lines.append("- " + _phrase("no longer may", f))
        lines.append(
            f"old_footprint {self.old_fingerprint}  new_footprint {self.new_fingerprint}"
        )
        payload = {
            "ok": self.ok,
            "old_fingerprint": self.old_fingerprint,
            "new_fingerprint": self.new_fingerprint,
            "added": sorted([f.subject, f.kind, f.ref, f.scope] for f in self.added),
            "removed": sorted([f.subject, f.kind, f.ref, f.scope] for f in self.removed),
        }
        lines.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return "\n".join(lines)


def authority_diff(old: PipelineConfig, new: PipelineConfig) -> AuthorityDiff:
    fo = effective_authority(old)
    fn = effective_authority(new)
    return AuthorityDiff(
        old_fingerprint=fo.fingerprint(),
        new_fingerprint=fn.fingerprint(),
        added=frozenset(fn.facts - fo.facts),
        removed=frozenset(fo.facts - fn.facts),
    )


# ---------------------------------------------------------------------------
# CI gate (Task 4)
# ---------------------------------------------------------------------------

def check_no_new_authority(old: PipelineConfig, new: PipelineConfig) -> None:
    """Raise AuthorityRegressionError if ``new`` grants any new reachable authority
    over ``old``; return None on identity or a pure narrowing. The CI-gate entrypoint."""
    diff = authority_diff(old, new)
    if not diff.ok:
        raise AuthorityRegressionError(diff.certificate(), diff=diff)
