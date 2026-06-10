# src/drawbore/testing/gauntlet/cases.py
"""Containment verdicts and the canonical attack-case corpus.

A ContainmentCase declares an attack `kind`, its `target` step, a kind-specific
`payload`, and the `expect`ed refusal verdict. The verdicts are derived from the
runtime's REAL halt taxonomy (see the runner's mapping), not the attack's name.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class Containment(str, Enum):
    SCHEMA_REJECT = "schema_reject"      # halt; output/input schema violation
    ESCALATED = "escalated"              # a confidence/approval trigger stopped the run (sync policy)
    FAIL_CLOSED = "fail_closed"          # halt; "testing_error" (unmocked external)
    MODEL_REFUSED = "model_refused"      # halt; "model_error" (loop refused: >1 call/turn or non-JSON final)
    DENIED_BREAKER = "denied_breaker"    # a proxied call denied at the circuit breaker
    DENIED_SCOPE = "denied_scope"        # a proxied call denied at operation/scope
    DENIED_TOKEN = "denied_token"        # a proxied call denied at token validation
    HALTED = "halted"                    # halted for another declared reason


@dataclass(frozen=True)
class ContainmentCase:
    name: str
    kind: Literal["schema_violation", "low_confidence", "unmocked_tool",
                  "breaker_trip", "parallel_tool_calls", "non_json_final"]
    target: str
    payload: Any
    expect: Containment


def schema_violation(target: str, bad_output: dict) -> ContainmentCase:
    if not isinstance(bad_output, dict):
        raise ValueError("schema_violation(bad_output) must be a dict that violates the output schema")
    return ContainmentCase(f"schema_violation:{target}", "schema_violation", target, bad_output, Containment.SCHEMA_REJECT)


def low_confidence(target: str, low_output: dict) -> ContainmentCase:
    if not isinstance(low_output, dict):
        raise ValueError("low_confidence(low_output) must be a dict (a below-threshold HasConfidence output)")
    return ContainmentCase(f"low_confidence:{target}", "low_confidence", target, low_output, Containment.ESCALATED)


def unmocked_tool(target: str, tool_ref: str) -> ContainmentCase:
    if not tool_ref:
        raise ValueError("unmocked_tool requires a tool_ref")
    return ContainmentCase(f"unmocked_tool:{target}", "unmocked_tool", target, tool_ref, Containment.FAIL_CLOSED)


def breaker_trip(target: str, tool_ref: str) -> ContainmentCase:
    if not tool_ref:
        raise ValueError("breaker_trip requires a tool_ref")
    return ContainmentCase(f"breaker_trip:{target}", "breaker_trip", target, tool_ref, Containment.DENIED_BREAKER)


def parallel_tool_calls(target: str, tool_a: str, tool_b: str) -> ContainmentCase:
    if not tool_a or not tool_b:
        raise ValueError("parallel_tool_calls requires two tool refs")
    return ContainmentCase(f"parallel_tool_calls:{target}", "parallel_tool_calls", target, (tool_a, tool_b), Containment.MODEL_REFUSED)


def non_json_final(target: str) -> ContainmentCase:
    return ContainmentCase(f"non_json_final:{target}", "non_json_final", target, None, Containment.MODEL_REFUSED)
