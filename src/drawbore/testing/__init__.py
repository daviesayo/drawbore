"""Local test mode — a scoped harness around real ``Pipeline.run``.

Test mode fakes the OUTSIDE WORLD (tools, model responses, loop turns); it never
fakes Drawbore. Every safety control — schema gates, context isolation, proxy/JIT,
circuit breakers, evidence policy, audit, escalation, identity, the agentic tool loop,
and config resolution — runs for real. ``pipeline.test_mode(...)`` is the only entrypoint.
"""

from .authority import assert_footprint_sound
from .credentials import StaticCredentialChecker
from .errors import TestingError
from .harness import TestMode, TestPipeline
from .loop import call, final, multi_call, text
from .models import LoopScript, ModelMock, ToolMock
from .gauntlet import (
    Containment, ContainmentCase,
    assert_contained, run_containment, run_pack,
    breaker_trip, low_confidence, non_json_final,
    parallel_tool_calls, schema_violation, unmocked_tool,
)

__all__ = [
    "assert_footprint_sound",
    "TestingError",
    "TestMode", "TestPipeline",
    "StaticCredentialChecker",
    "ToolMock", "ModelMock", "LoopScript",
    "call", "final", "text", "multi_call",
    "Containment", "ContainmentCase",
    "run_containment", "assert_contained", "run_pack",
    "schema_violation", "low_confidence", "unmocked_tool",
    "breaker_trip", "parallel_tool_calls", "non_json_final",
]
