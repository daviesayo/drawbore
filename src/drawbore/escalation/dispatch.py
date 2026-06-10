"""Escalation delivery.

The MIT core defines the dispatcher interface and an in-process recorder. Real
channels (Slack, email, webhook, SMS) and the approve/reject interface are
managed-service (out of scope for the open-source core).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .package import EscalationPackage
from .policy import EscalationPolicy


class EscalationDispatcher(ABC):
    @abstractmethod
    def dispatch(self, package: EscalationPackage, policy: EscalationPolicy) -> None:
        """Deliver ``package`` to ``policy``'s channel/target."""


class RecordingDispatcher(EscalationDispatcher):
    """In-process dispatcher that records what would be sent — the Phase-1 default
    for local runs and tests."""

    def __init__(self) -> None:
        self.sent: list[tuple[EscalationPackage, EscalationPolicy]] = []

    def dispatch(self, package: EscalationPackage, policy: EscalationPolicy) -> None:
        self.sent.append((package, policy))
