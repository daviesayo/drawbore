"""Per-step resume seal: the declared semantics a checkpoint was taken under.

A completed step's seal is recorded alongside its checkpointed output. On
resume, every stored seal is re-verified against the live pipeline BEFORE any
step executes; a mismatch refuses the resume rather than replaying outputs
into drifted logic.

The seal covers DECLARED semantics (the agent's contract fields and the
step's evidence policy), not Python function bodies — changing an agent's
implementation without bumping ``version`` is undetectable by design, and
bumping ``version`` on behavior change is the author's responsibility.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from drawbore._canon import canonical_fingerprint, text_fingerprint

#: Human-facing labels for sealed fields whose model name is an internal
#: fingerprint column. Halt reasons and ledger prose use these; the
#: machine-facing drifted-fields tuples keep the model field names.
FIELD_LABELS: dict[str, str] = {
    "instructions_fingerprint": "instructions",
    "input_schema_fingerprint": "input schema",
    "output_schema_fingerprint": "output schema",
    "evidence_policy_fingerprint": "evidence policy",
}


def field_label(field: str) -> str:
    """The human-facing label for a sealed field name."""
    return FIELD_LABELS.get(field, field)


class StepSeal(BaseModel):
    """Frozen snapshot of one step's declared semantics at checkpoint time.

    Scalar contract fields are stored verbatim (so drift reports can show
    values); bulky fields are stored as canonical SHA-256 fingerprints.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent: str
    version: str
    risk_tier: str
    requires_human_approval: bool
    context_access: str
    tools: tuple[str, ...]
    model: str | None
    fallback_model: str | None
    instructions_fingerprint: str
    input_schema_fingerprint: str
    output_schema_fingerprint: str
    evidence_policy_fingerprint: str


def seal_for(spec: "AgentSpec", evidence_policy: "EvidencePolicy | None") -> StepSeal:
    """Build the seal for a step from its agent contract and evidence policy."""
    if evidence_policy is None:
        evidence_fp = "none"
    else:
        evidence_fp = canonical_fingerprint(evidence_policy.model_dump(mode="json"))
    return StepSeal(
        agent=spec.name,
        version=spec.version,
        risk_tier=spec.risk_tier,
        requires_human_approval=spec.requires_human_approval,
        context_access=spec.context_access,
        tools=tuple(spec.tools),
        model=spec.model,
        fallback_model=spec.fallback_model,
        instructions_fingerprint=text_fingerprint(spec.instructions),
        input_schema_fingerprint=canonical_fingerprint(spec.input.model_json_schema()),
        output_schema_fingerprint=canonical_fingerprint(spec.output.model_json_schema()),
        evidence_policy_fingerprint=evidence_fp,
    )


def diff_seals(stored: StepSeal, current: StepSeal) -> tuple[str, ...]:
    """Sorted tuple of drifted field names; empty when sealed semantics match."""
    drifted = [
        name
        for name in StepSeal.model_fields
        if getattr(stored, name) != getattr(current, name)
    ]
    return tuple(sorted(drifted))
