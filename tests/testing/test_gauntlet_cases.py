# tests/testing/test_gauntlet_cases.py
import pytest

from drawbore.testing.gauntlet.cases import (
    Containment, ContainmentCase,
    schema_violation, low_confidence, unmocked_tool,
    breaker_trip, parallel_tool_calls, non_json_final,
)


def test_builders_set_kind_target_payload_and_expect():
    assert schema_violation("scorer", {"x": 1}) == ContainmentCase(
        "schema_violation:scorer", "schema_violation", "scorer", {"x": 1}, Containment.SCHEMA_REJECT)
    assert low_confidence("scorer", {"confidence": 0.1}).expect is Containment.ESCALATED
    assert unmocked_tool("rev", "t:a").expect is Containment.FAIL_CLOSED
    assert breaker_trip("rev", "t:a").expect is Containment.DENIED_BREAKER
    assert parallel_tool_calls("rev", "t:a", "t:b").payload == ("t:a", "t:b")
    assert parallel_tool_calls("rev", "t:a", "t:b").expect is Containment.MODEL_REFUSED
    assert non_json_final("rev").expect is Containment.MODEL_REFUSED


def test_builders_validate_their_bindings_eagerly():
    with pytest.raises(ValueError):
        schema_violation("scorer", ["not", "a", "dict"])   # bad_output must be a dict
    with pytest.raises(ValueError):
        low_confidence("scorer", "nope")
    with pytest.raises(ValueError):
        breaker_trip("rev", "")
    with pytest.raises(ValueError):
        unmocked_tool("rev", "")
    with pytest.raises(ValueError):
        parallel_tool_calls("rev", "t:a", "")
    with pytest.raises(ValueError):
        parallel_tool_calls("rev", "", "t:b")


def test_gauntlet_surface_is_exported_from_drawbore_testing():
    from drawbore.testing import (
        Containment, ContainmentCase, run_containment, assert_contained, run_pack,
        schema_violation, low_confidence, unmocked_tool,
        breaker_trip, parallel_tool_calls, non_json_final,
    )
    assert Containment.SCHEMA_REJECT.value == "schema_reject"
