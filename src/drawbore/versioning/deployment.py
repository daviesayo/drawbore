"""Deployment staging + rollback.

In a regulated context you never big-bang deploy: shadow first (breaking changes),
canary second (non-breaking), full rollout only after the required stages pass; any
gate failure routes back to rollback, not forward. ``RolloutPlan`` enforces the
stage ordering; ``Deployment`` tracks the last stable version and keeps rollback
always available. Live traffic-splitting and the rollback dashboard are
managed-service (out of scope for the open-source core) — this is the in-core
ordering/state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from drawbore.errors import DrawboreError

from .compat import ChangeKind

_STAGES: dict[ChangeKind, tuple[str, ...]] = {
    "breaking": ("shadow", "canary", "full"),
    "non_breaking": ("canary", "full"),
}


class DeploymentError(DrawboreError):
    """Raised on an illegal rollout transition (skipping a stage, advancing an
    unpassed stage, completing an unfinished rollout, or beginning a new rollout
    while one is already in flight)."""


@dataclass
class RolloutPlan:
    """An in-flight rollout of ``new_version``. Breaking changes traverse
    shadow→canary→full; non-breaking changes canary→full. A stage must be marked
    passed before the rollout can advance to the next stage."""

    change_kind: ChangeKind
    new_version: str
    _index: int = field(default=0, init=False)
    _passed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.change_kind not in _STAGES:
            raise DeploymentError(f"unknown change_kind: {self.change_kind!r}")

    @property
    def stages(self) -> tuple[str, ...]:
        return _STAGES[self.change_kind]

    @property
    def stage(self) -> str:
        return self.stages[self._index]

    @property
    def is_complete(self) -> bool:
        return self._index == len(self.stages) - 1 and self._passed

    def mark_passed(self) -> None:
        """Record that the current stage's gate passed."""
        self._passed = True

    def advance(self) -> None:
        """Move to the next stage. Fails closed if the current stage has not
        passed, or if already at the final stage."""
        if not self._passed:
            raise DeploymentError(
                f"stage '{self.stage}' has not passed; cannot advance"
            )
        if self._index == len(self.stages) - 1:
            raise DeploymentError("rollout is already at its final stage")
        self._index += 1
        self._passed = False


@dataclass
class Deployment:
    """Tracks the last stable version and the in-flight rollout (if any).
    Rollback to the last stable version is always available."""

    stable_version: str
    in_flight: RolloutPlan | None = field(default=None)

    def begin(self, plan: RolloutPlan) -> None:
        """Start a rollout. A ``RolloutPlan`` is single-use: it must be fresh
        (no stage advanced or passed), so a completed/advanced plan cannot be
        re-begun to bypass the staging gates. A new rollout cannot be started while
        one is already in flight — call ``rollback()`` (or ``complete()``) first, so
        an in-flight rollout is never silently abandoned."""
        if self.in_flight is not None:
            raise DeploymentError(
                "a rollout is already in flight; call rollback() before beginning a new one"
            )
        if plan._index != 0 or plan._passed:
            raise DeploymentError(
                "begin requires a fresh RolloutPlan (one plan, one rollout)"
            )
        self.in_flight = plan

    def rollback(self) -> str:
        """Atomically discard the in-flight rollout and return to the last stable
        version. Always available — a failed gate routes here, not forward."""
        self.in_flight = None
        return self.stable_version

    def complete(self) -> None:
        """Promote the in-flight version to stable. Fails closed unless the rollout
        has traversed every required stage."""
        if self.in_flight is None or not self.in_flight.is_complete:
            raise DeploymentError(
                "rollout is not complete; full rollout requires every stage to pass"
            )
        self.stable_version = self.in_flight.new_version
        self.in_flight = None
