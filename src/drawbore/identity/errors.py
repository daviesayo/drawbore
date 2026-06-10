"""Identity error type."""

from __future__ import annotations

from drawbore.errors import DrawboreError


class IdentityError(DrawboreError):
    """Raised when an identity rule is violated — missing sponsor, illegal
    lifecycle transition, or an unauthorised re-attestation."""
