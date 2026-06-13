# src/drawbore/testing/gauntlet/runner.py
"""Run a containment case through test_mode and map the result to a verdict.

The verdict is read from the public RunResult ONLY. A stop surfaces as
status=="halted" OR (with an on_failure policy — the framework default)
status=="escalated"; either way the refusal CLASS is resolved from the reason +
audit FIRST, and ESCALATED is the fallback only for a confidence/approval trigger.
"""

from __future__ import annotations

from drawbore.tools.proxy import DEFAULT_MAX_CALLS_PER_TOOL

from ..loop import call, final, multi_call, text
from .cases import Containment, ContainmentCase

_DENIAL = {
    "denied:breaker": Containment.DENIED_BREAKER,
    "denied:scope": Containment.DENIED_SCOPE,
    "denied:token": Containment.DENIED_TOKEN,
}


def _first_denial(audit) -> "Containment | None":
    """First `-> denied:<x>` label in the audit's tool_calls (by step order). The
    render format is `"<tool> (<op>) -> <result>"` (pipeline/executor.py:_tool_calls_since)."""
    if audit is None:
        return None
    for step in audit.step_records:
        for call_str in step.tool_calls:
            label = call_str.rsplit("-> ", 1)[-1]
            if label in _DENIAL:
                return _DENIAL[label]
    return None


def _verdict(result) -> "Containment | None":
    """Map a RunResult to a Containment, or None if the attack was NOT contained
    (the run COMPLETED). Resolves the refusal class before the ESCALATED fallback so
    an on_failure policy (which re-labels hard halts as "escalated") doesn't mask it."""
    if result.status == "completed":
        return None
    denied = _first_denial(result.audit_trace)
    if denied is not None:
        return denied
    audit = result.audit_trace
    if audit is not None and getattr(audit, "schema_violations", 0) >= 1:
        return Containment.SCHEMA_REJECT
    reason = result.reason or ""
    # reason is prefix-tokened by executor.py:161 (f"{halt_reason_for(exc)}: {exc}").
    if reason.startswith("testing_error:"):
        return Containment.FAIL_CLOSED
    if reason.startswith("model_error:"):
        return Containment.MODEL_REFUSED
    if result.status == "escalated":
        return Containment.ESCALATED
    return Containment.HALTED


_BREAKER_MAX: int = DEFAULT_MAX_CALLS_PER_TOOL

_BENIGN_FINAL: dict = {}            # never reached by the loop attack cases (they abort first)
_BENIGN_TOOL_RESULT: dict = {"ok": True}

_MOCK_KEYS = ("mock_model_responses", "mock_loop_scripts", "mock_tools")


def _provoke(case: ContainmentCase) -> dict:
    """The per-kind test_mode mock bundle that provokes the attack on case.target."""
    k = case.kind
    if k == "schema_violation" or k == "low_confidence":
        return {"mock_model_responses": {case.target: case.payload}}
    if k == "unmocked_tool":
        # call the tool but DON'T mock it -> fail-closed handler -> testing_error
        return {"mock_loop_scripts": {case.target: [call(case.payload), final(_BENIGN_FINAL)]}}
    if k == "breaker_trip":
        ref = case.payload
        script = [call(ref) for _ in range(_BREAKER_MAX + 1)] + [final(_BENIGN_FINAL)]
        return {"mock_loop_scripts": {case.target: script},
                "mock_tools": {ref: _BENIGN_TOOL_RESULT}}
    if k == "parallel_tool_calls":
        a, b = case.payload
        return {"mock_loop_scripts": {case.target: [multi_call(call(a), call(b))]}}
    if k == "non_json_final":
        # A non-JSON loop turn earns ONE bounded reprompt for a structured response, so
        # provoke a guaranteed refusal with non-JSON on BOTH the original turn and the
        # reprompt — the model never produces a tool call or a JSON answer.
        return {"mock_loop_scripts": {case.target: [text("not json"), text("still not json")]}}
    raise ValueError(f"unknown containment case kind {k!r}")


def _merge(baseline: "dict | None", provocation: dict) -> dict:
    """Merge benign baseline mocks (non-target steps) with the case's provocation."""
    merged: dict = {key: {} for key in _MOCK_KEYS}
    for src in (baseline or {}, provocation):
        for key in _MOCK_KEYS:
            merged[key].update(src.get(key, {}))
    return {key: val for key, val in merged.items() if val}


async def run_containment(
    pipeline,
    case: ContainmentCase,
    *,
    initial,
    baseline=None,
    llm_config=None,
    credential_checker=None,
):
    """Provoke `case` on `pipeline` through test_mode; return the observed Containment,
    or None if the run COMPLETED (the attack was NOT contained).

    Pass `llm_config` / `credential_checker` for pipelines whose agents bind a
    `model="profile:..."` reference; they are forwarded to `test_mode` so the profile
    resolves (against the fake credential checker) instead of halting `model_config_error`
    before the attack is provoked."""
    mocks = _merge(baseline, _provoke(case))
    async with pipeline.test_mode(
        **mocks, llm_config=llm_config, credential_checker=credential_checker
    ) as tp:
        result = await tp.run(initial)
    return _verdict(result)


async def assert_contained(
    pipeline,
    case: ContainmentCase,
    *,
    initial,
    baseline=None,
    llm_config=None,
    credential_checker=None,
) -> None:
    observed = await run_containment(
        pipeline,
        case,
        initial=initial,
        baseline=baseline,
        llm_config=llm_config,
        credential_checker=credential_checker,
    )
    if observed != case.expect:
        raise AssertionError(
            f"containment case {case.name!r} ({case.kind}): expected {case.expect}, got {observed}"
        )


async def run_pack(
    pipeline,
    cases,
    *,
    initial,
    baseline=None,
    llm_config=None,
    credential_checker=None,
) -> dict:
    """Run every case (all targeting steps in `pipeline`); return {case.name: observed}."""
    report: dict = {}
    for case in cases:
        report[case.name] = await run_containment(
            pipeline,
            case,
            initial=initial,
            baseline=baseline,
            llm_config=llm_config,
            credential_checker=credential_checker,
        )
    return report
