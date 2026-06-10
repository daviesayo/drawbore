"""Structural data-trust labels and the per-run taint ledger.

A two-point lattice (TRUSTED < UNTRUSTED, UNTRUSTED absorbing) carried on every
pipeline value. The TaintLedger holds the per-(run, step) trust scope the proxy
gates exfil-capable calls against. Pure leaf: stdlib only, no drawbore imports.
"""

from __future__ import annotations

from enum import Enum


class TrustLabel(str, Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


def join(*labels: TrustLabel) -> TrustLabel:
    """The lattice join: UNTRUSTED if any argument is UNTRUSTED, else TRUSTED.

    Commutative, associative, idempotent; TRUSTED is the identity (join() is TRUSTED).
    """
    for label in labels:
        if label is TrustLabel.UNTRUSTED:
            return TrustLabel.UNTRUSTED
    return TrustLabel.TRUSTED


class TaintLedger:
    """Per-run mutable map of (run_id, step) -> trust scope.

    Seeded by the run loop at step entry; widened by the proxy when an
    UNTRUSTED-source tool is invoked. ``managed=True`` (the pipeline's run ledger)
    makes an unseeded-but-executed step fail closed to UNTRUSTED; a bare ledger
    (standalone proxy, e.g. a unit test) defaults unseeded steps to TRUSTED —
    but observes still taint and still gate within a (run, step).

    ``step`` may be ``None`` (the proxy uses ``getattr(run_ctx, "step", None)``);
    ``None`` is a valid key and fails closed under a managed ledger.

    ``initial_trust`` does not affect ``scope()``'s default; it is advisory state
    the pipeline run loop reads to seed the first step (exposed via ``_initial``).
    """

    def __init__(
        self,
        initial_trust: TrustLabel = TrustLabel.TRUSTED,
        *,
        managed: bool = False,
    ) -> None:
        self._initial = initial_trust            # the run loop seeds the first step from this
        self._managed = managed
        self._scope: dict[tuple[str, int | None], TrustLabel] = {}

    def _default(self) -> TrustLabel:
        return TrustLabel.UNTRUSTED if self._managed else TrustLabel.TRUSTED

    def seed(self, run_id: str, step: int | None, label: TrustLabel) -> None:
        """Set the trust scope for a (run_id, step) pair.

        Monotone: if a scope has already been recorded for this key (e.g. by a
        prior ``observe`` call), the result is the lattice join of the existing
        label and the new one. A seed call can never lower an already-observed
        scope toward TRUSTED.
        """
        key = (run_id, step)
        existing = self._scope.get(key)
        self._scope[key] = label if existing is None else join(existing, label)

    def observe(self, run_id: str, step: int | None, source_trust: TrustLabel) -> None:
        """Monotonically widen the taint scope for (run_id, step) toward UNTRUSTED.

        The join can only move the scope toward UNTRUSTED in the lattice; it can
        never move back to TRUSTED once tainted.
        """
        base = self._scope.get((run_id, step), self._default())
        self._scope[(run_id, step)] = join(base, source_trust)

    def scope(self, run_id: str, step: int | None) -> TrustLabel:
        """Return the current trust scope for (run_id, step)."""
        return self._scope.get((run_id, step), self._default())
