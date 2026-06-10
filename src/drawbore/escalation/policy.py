"""Escalation policy — where and how a triggered escalation is delivered."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EscalationPolicy:
    """Declares an escalation's destination and blocking mode.

    - ``mode="sync"``: a synchronous gate — the run halts and the human must act
      (irreversible actions; the run's status becomes ``"escalated"``).
    - ``mode="async"``: an asynchronous review — the package is dispatched but the
      run continues (reviewable actions, e.g. flag for audit). In Phase 1 this is
      honoured for the confidence trigger; error halts and approval gates always
      block regardless of mode.
    """

    channel: str
    target: str
    mode: Literal["sync", "async"] = "sync"
