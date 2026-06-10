"""Agent identity and its attestation surface.

Every agent has an identity, not just a name. The identity is the immutable
record issued at registration; the mutable lifecycle state lives on the registry
record (see ``registry.py``). The *attestation surface* is the fingerprint whose
change forces re-attestation by the human sponsor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

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


def _schema_fingerprint(model: type[BaseModel]) -> str:
    """A structural fingerprint of a model for re-attestation.

    Uses Pydantic's JSON schema, which recursively expands nested models, so a
    structural change anywhere in the (possibly nested) schema — a field added or
    removed, a type changed, a required-ness flipped — changes the fingerprint and
    forces re-attestation. ``model_json_schema()`` is the correct tool for
    change-detection evidence (it is only barred as the type-assignability oracle,
    which this is not). ``sort_keys`` makes the serialization deterministic.
    """
    return json.dumps(model.model_json_schema(), sort_keys=True)


def attestation_surface(spec) -> tuple:
    """Compute the re-attestation surface of an agent spec: the tuple of
    (tool declarations, input schema, output schema, risk tier). A change
    to ANY of these — including a nested schema change — requires re-attestation by
    the human sponsor.

    ``spec`` is a ``drawbore.agent.AgentSpec``; typed loosely to keep ``identity``
    a leaf with no import of ``agent``.
    """
    return (
        tuple(sorted(spec.tools)),
        _schema_fingerprint(spec.input),
        _schema_fingerprint(spec.output),
        spec.risk_tier,
    )
