"""Versioning-specific errors."""

from drawbore.errors import DrawboreError


class DeploymentError(DrawboreError):
    """Raised on an illegal rollout transition (skipping a stage, advancing an
    unpassed stage, completing an unfinished rollout, or beginning a new rollout
    while one is already in flight)."""
