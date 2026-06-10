"""Footprint soundness cross-check. A test-only helper: assert the runtime
exercised no tool/evidence ref outside the manifest's static footprint. Reads the
public ``RunResult.audit_trace`` (no runtime seam). The value is as a regression
tripwire — it fails the day a future feature binds a ref the manifest cannot see.
"""

from __future__ import annotations

from drawbore.config import effective_authority


def exercised_refs(audit_trace) -> frozenset[str]:
    """The set of tool/evidence refs the runtime actually called, parsed from the
    audit trace's ``tool_calls`` (rendered ``"<tool> (<op>) -> <result>"`` by
    ``pipeline/executor.py:_tool_calls_since``)."""
    refs: set[str] = set()
    for step in audit_trace.step_records:
        for call in step.tool_calls:
            refs.add(call.split(" (", 1)[0])
    return frozenset(refs)


def assert_footprint_sound(config, audit_trace) -> None:
    """Raise AssertionError if the run exercised any ref absent from the static
    footprint of ``config`` (compared at ref granularity)."""
    static = {f.ref for f in effective_authority(config).facts}
    extra = exercised_refs(audit_trace) - static
    if extra:
        raise AssertionError(
            "runtime exercised authority outside the static footprint: "
            f"{sorted(extra)} (footprint refs: {sorted(static)})"
        )
