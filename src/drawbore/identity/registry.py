"""In-process agent identity registry.

The MIT core ships an in-memory registry that enforces the identity invariants:
no agent without a human sponsor, a strict ``draft → active → suspended →
decommissioned`` lifecycle, single-action atomic decommission, and a re-attestation
block when the attestation surface changes. Durable, multi-tenant identity storage
is managed-service (out of scope for the open-source core); it backs this same surface.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from .errors import IdentityError
from .identity import AgentIdentity, LifecycleState, attestation_surface


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _Record:
    identity: AgentIdentity
    state: LifecycleState
    attested_surface: tuple
    needs_reattestation: bool = False
    pending_surface: tuple | None = None


class IdentityRegistry:
    """Registers agents by name and enforces the identity lifecycle + re-attestation."""

    def __init__(self) -> None:
        self._by_name: dict[str, _Record] = {}

    # --- registration (JIT issuance) -------------------------------------
    def register(
        self,
        spec,
        *,
        sponsor: str,
        purpose: str,
        ttl_seconds: float,
        pipeline: str | None = None,
        tenant: str | None = None,
        agent_id: str | None = None,
        now=None,
    ) -> AgentIdentity:
        """Issue an identity for ``spec`` and move it to ``active``. A human
        sponsor is mandatory: no agent exists without an owner."""
        if not sponsor or not sponsor.strip():
            raise IdentityError(
                "a human sponsor is required: no agent exists without an owner"
            )
        if spec.name in self._by_name:
            raise IdentityError(f"agent '{spec.name}' is already registered")
        identity = AgentIdentity(
            agent_id=agent_id or f"{spec.name}-{uuid.uuid4().hex[:12]}",
            name=spec.name,
            purpose=purpose,
            risk_tier=spec.risk_tier,
            sponsor=sponsor,
            ttl_seconds=ttl_seconds,
            created_at=(now or _utc_now)(),
            pipeline=pipeline,
            tenant=tenant,
        )
        self._by_name[spec.name] = _Record(
            identity=identity,
            state="active",
            attested_surface=attestation_surface(spec),
        )
        return identity

    # --- lifecycle transitions -------------------------------------------
    def suspend(self, name: str) -> None:
        rec = self._require(name)
        if rec.state != "active":
            raise IdentityError(f"cannot suspend '{name}' from state '{rec.state}'")
        rec.state = "suspended"

    def resume(self, name: str) -> None:
        rec = self._require(name)
        if rec.state != "suspended":
            raise IdentityError(f"cannot resume '{name}' from state '{rec.state}'")
        rec.state = "active"

    def decommission(self, name: str) -> None:
        """Single atomic action: revoke and mark inactive, terminally."""
        rec = self._require(name)
        rec.state = "decommissioned"
        rec.needs_reattestation = False
        rec.pending_surface = None

    # --- re-attestation ---------------------------------------------------
    def update(self, spec) -> None:
        """Re-evaluate an agent's attestation surface after a redefinition. If the
        surface changed, record it as the pending surface and block the agent until
        the sponsor re-attests against exactly that surface."""
        rec = self._require(spec.name)
        if rec.state == "decommissioned":
            raise IdentityError("cannot update a decommissioned agent")
        surface = attestation_surface(spec)
        if surface != rec.attested_surface:
            rec.needs_reattestation = True
            rec.pending_surface = surface

    def reattest(self, name: str, *, sponsor: str, spec) -> None:
        """Approve the pending surface change. Must be the agent's human sponsor,
        and the supplied ``spec`` must match the surface that triggered the block —
        so a privilege escalation cannot be laundered by re-attesting against a
        different (e.g. lower-risk) surface."""
        rec = self._require(name)
        if rec.state == "decommissioned":
            raise IdentityError("cannot re-attest a decommissioned agent")
        if sponsor != rec.identity.sponsor:
            raise IdentityError(
                "re-attestation must be approved by the human sponsor"
            )
        if rec.pending_surface is None or attestation_surface(spec) != rec.pending_surface:
            raise IdentityError(
                "re-attestation spec must match the change that triggered the block"
            )
        rec.attested_surface = rec.pending_surface
        rec.pending_surface = None
        rec.needs_reattestation = False

    # --- queries (the pipeline gate uses these) --------------------------
    def is_registered(self, name: str) -> bool:
        return name in self._by_name

    def state_of(self, name: str) -> LifecycleState:
        rec = self._by_name.get(name)
        return rec.state if rec is not None else "draft"

    def agent_id_of(self, name: str) -> str | None:
        rec = self._by_name.get(name)
        return rec.identity.agent_id if rec is not None else None

    def run_block_reason(self, name: str) -> str | None:
        """``None`` if the agent may run; otherwise a legible reason. An
        unregistered (draft) agent is never blocked here — draft is the
        schema-only path."""
        rec = self._by_name.get(name)
        if rec is None:
            return None
        if rec.state == "decommissioned":
            return "decommissioned"
        if rec.state == "suspended":
            return "suspended"
        if rec.needs_reattestation:
            return "reattestation_required"
        return None

    def _require(self, name: str) -> _Record:
        rec = self._by_name.get(name)
        if rec is None:
            raise IdentityError(f"agent '{name}' is not registered")
        return rec
