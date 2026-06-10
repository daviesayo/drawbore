# src/drawbore/testing/gauntlet/__init__.py
"""The invariant safety gauntlet: a reusable containment corpus.

Drives canonical agent attacks through the live test_mode safety layer and asserts
the runtime refused. Public surface: the six case builders, the Containment verdicts,
and run_containment / assert_contained / run_pack.
"""

from .cases import (
    Containment, ContainmentCase,
    schema_violation, low_confidence, unmocked_tool,
    breaker_trip, parallel_tool_calls, non_json_final,
)
from .runner import run_containment, assert_contained, run_pack

__all__ = [
    "Containment", "ContainmentCase",
    "schema_violation", "low_confidence", "unmocked_tool",
    "breaker_trip", "parallel_tool_calls", "non_json_final",
    "run_containment", "assert_contained", "run_pack",
]
