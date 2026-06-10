"""Run-scoped orchestrator metadata.

This object is orchestrator-only. Agents never receive or access it — that
isolation is what keeps stateless agents easy to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunState:
    run_id: str
    step: int = 0
    error_count: int = 0
