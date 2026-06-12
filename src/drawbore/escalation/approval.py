"""Typed human-approval artifacts.

An ``ApprovalRequest`` is what the runtime asks a human; an ``ApprovalDecision``
is the typed answer the runtime applies. The request carries the escalation's
WHY as legible TEXT (never the package's arbitrarily-typed payload fields, which
are not JSON-mode safe) plus the step's validated output as a plain dict, so
both artifacts round-trip through JSON. Reviewer identity is an opaque,
caller-supplied string — agent identity and human reviewer auth are different
concerns and are never conflated.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ApprovalRequest(BaseModel):
    """What a human is being asked to approve, as a self-contained artifact."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    run_id: str
    step: str
    question: str
    reason: str
    package_legible: str
    proposed_output: dict[str, Any]


class ApprovalDecision(BaseModel):
    """A single-use, request-bound human decision."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    verdict: Literal["approved", "rejected", "amended"]
    reviewer_id: str
    rationale: str | None = None
    amended_output: dict[str, Any] | None = None

    @field_validator("reviewer_id")
    @classmethod
    def _reviewer_named(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("reviewer_id must be a non-blank reviewer identity")
        return value

    @model_validator(mode="after")
    def _amendment_matches_verdict(self) -> "ApprovalDecision":
        if self.verdict == "amended" and self.amended_output is None:
            raise ValueError("verdict 'amended' requires amended_output")
        if self.verdict != "amended" and self.amended_output is not None:
            raise ValueError("amended_output is only valid with verdict 'amended'")
        return self
