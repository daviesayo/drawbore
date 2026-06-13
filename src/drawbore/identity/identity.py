"""Agent identity and its attestation surface.

Every agent has an identity, not just a name. The identity is the immutable
record issued at registration; the mutable lifecycle state lives on the registry
record (see ``registry.py``). The *attestation surface* is the fingerprint whose
change forces re-attestation by the human sponsor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from drawbore._canon import canonical_fingerprint

LifecycleState = Literal["draft", "active", "suspended", "decommissioned"]
RiskTier = Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True)
class AgentIdentity:
    """The unique identity issued to a registered agent.

    Attributes: a unique id, a TTL, the declared purpose, the risk tier, the
    delegation context (pipeline + tenant), the always-required human sponsor,
    and a creation timestamp.
    """

    agent_id: str
    name: str
    purpose: str
    risk_tier: RiskTier
    sponsor: str
    ttl_seconds: float
    created_at: str
    pipeline: str | None = None
    tenant: str | None = None


def attestation_surface(spec) -> tuple:
    """Compute the re-attestation surface of an agent spec: the tuple of
    (tool declarations, input schema, output schema, risk tier). A change
    to ANY of these — including a nested schema change — requires re-attestation by
    the human sponsor.

    ``spec`` is a ``drawbore.agent.AgentSpec``; typed loosely to keep ``identity``
    a leaf with no import of ``agent``.

    Uses ``model_json_schema()`` which recursively expands nested models, so a
    structural change anywhere in the (possibly nested) schema forces re-attestation.
    ``canonical_fingerprint`` applies ``sort_keys=True`` for deterministic output.
    """
    return (
        tuple(sorted(spec.tools)),
        canonical_fingerprint(spec.input.model_json_schema()),
        canonical_fingerprint(spec.output.model_json_schema()),
        spec.risk_tier,
    )
