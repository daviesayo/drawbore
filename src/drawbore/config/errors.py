"""Config-layer error.

Self-declares ``halt_reason`` so a config-resolution failure escalates legibly
without ``drawbore.errors`` importing ``drawbore.config``."""

from __future__ import annotations

from drawbore.errors import DrawboreError


class ConfigResolutionError(DrawboreError):
    """A JSON config manifest could not be resolved into a valid pipeline — a
    missing/ambiguous agent ref, declaration drift, an invalid tagged-input mode, a
    lying ``depends_on``, an unsupported declaration, an unknown field, or a
    static-compatibility failure surfaced from ``Pipeline.add``. Fail-closed: config
    loading never silently ignores any of these."""

    halt_reason = "config_resolution_error"


class AuthorityRegressionError(DrawboreError):
    """A pipeline edit expanded the statically reachable capability authority.
    Carries the ``AuthorityDiff`` so callers can print its ``certificate()``."""

    halt_reason = "authority_regression"

    def __init__(self, message: str, *, diff: object) -> None:
        super().__init__(message)
        self.diff = diff
