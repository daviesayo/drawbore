"""Authority monotonicity over manifests — a thin seam over the config layer's
effective-authority diff, named here so the gate's second layer reads as policy."""

from __future__ import annotations

from drawbore.config.authority import AuthorityDiff, authority_diff
from drawbore.config.models import PipelineConfig


def authority_delta(baseline: PipelineConfig, candidate: PipelineConfig) -> AuthorityDiff:
    """The reachable-capability delta of ``candidate`` over ``baseline``.
    ``delta.ok`` is False iff the candidate WIDENS authority — the gate never
    auto-admits a widening; it returns the diff for a human to review."""
    return authority_diff(baseline, candidate)
