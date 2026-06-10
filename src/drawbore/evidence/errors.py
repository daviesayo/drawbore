"""Evidence-compression errors.

These self-declare ``halt_reason`` so a fail-closed compression or retrieval
escalates with a legible reason (``"evidence_error"``) rather than the generic
``"agent_error"``, keeping ``drawbore.errors`` free of any ``drawbore.evidence``
import.
"""

from __future__ import annotations

from drawbore.errors import DrawboreError


class EvidenceError(DrawboreError):
    """Base class for evidence-layer failures. ``halt_reason`` makes a fail-closed
    evidence failure legible in an escalation."""

    halt_reason = "evidence_error"


class EvidenceStoreError(EvidenceError):
    """The original evidence could not be stored — compression must not proceed
    (originals are always retained)."""


class EvidenceTransformError(EvidenceError):
    """A transform produced output the layer cannot use."""


class EvidenceRetrievalError(EvidenceError):
    """A retrieval failed closed — unknown/expired handle, out-of-scope mode, or
    oversize. A handle is data, not authority."""
