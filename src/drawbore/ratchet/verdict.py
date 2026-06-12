"""The admission verdict: one frozen, legible record per ``admit()`` call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from drawbore.config.authority import AuthorityDiff
    from drawbore.pipeline import Pipeline

RejectionLayer = Literal["resolver", "authority", "corpus", "corpus_integrity"]


@dataclass(frozen=True)
class RatchetVerdict:
    """The outcome of one admission attempt.

    ``pipeline`` is the blessed, already-resolved object iff ``admitted`` --
    adoption is the caller replacing its reference with it. ``corpus_root_after``
    is set only when admission appended newly derived cases.
    """

    admitted: bool
    pipeline: "Pipeline | None"
    manifest_fingerprint: str
    rejection_layer: RejectionLayer | None
    authority_diff: "AuthorityDiff | None"
    failed_case_id: str | None
    failed_property_description: str | None
    corpus_root_before: str
    corpus_root_after: str | None
    reason: str | None = None

    def legible(self) -> str:
        """A regulator-readable, single-block account of the decision."""
        if self.admitted:
            lines = [f"Proposed manifest {self.manifest_fingerprint} ADMITTED."]
            if self.corpus_root_after is not None:
                lines.append(
                    f"Regression corpus root advanced from {self.corpus_root_before} "
                    f"to {self.corpus_root_after}."
                )
            else:
                lines.append(
                    f"Regression corpus root unchanged {self.corpus_root_before} "
                    f"(no new cases derived)."
                )
            return "\n".join(lines)
        lines = [
            f"Proposed manifest {self.manifest_fingerprint} REJECTED "
            f"at the {self.rejection_layer} layer."
        ]
        if self.failed_case_id is not None:
            lines.append(
                f"Frozen case {self.failed_case_id} failed under the real safety "
                f"layer: {self.failed_property_description}"
            )
        if self.authority_diff is not None:
            lines.append(self.authority_diff.certificate())
        if self.reason is not None:
            lines.append(self.reason)
        lines.append(f"Corpus root unchanged {self.corpus_root_before}.")
        return "\n".join(lines)
