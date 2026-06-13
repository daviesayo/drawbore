# tests/config/test_schema_oracle.py
"""Soundness battery for the directional input-schema relaxation oracle.

The cardinal property: a genuine input-schema relaxation (a strictly wider
accepted-input set) must NEVER pass. A false flag on a safe change is acceptable
(fail-closed). Schemas are built as plain dicts for precise control.
"""

import pytest

from drawbore.config import (
    SchemaRelaxation,
    SchemaRelaxationDiff,
    check_no_schema_relaxation,
    schema_relaxation_diff,
)
from drawbore.config.errors import SchemaRelaxationError
from drawbore.config.models import AgentConfig, PipelineConfig


def _agent(name: str, input_schema: dict) -> dict:
    return {
        "name": name,
        "ref": f"x.{name}",
        "version": "1.0",
        "risk_tier": "low",
        "requires_human_approval": False,
        "context_access": "none",
        "tools": [],
        "model": None,
        "fallback_model": None,
        "instructions": None,
        "input_schema": input_schema,
        "output_schema": {"type": "object"},
        "input_schema_hash": "sha256:0",
        "output_schema_hash": "sha256:0",
    }


def _cfg(input_schema: dict, *, name: str = "a") -> PipelineConfig:
    """A one-agent PipelineConfig carrying the given input schema."""
    return PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "pipeline": {"name": "p", "version": "1.0"},
            "agents": [_agent(name, input_schema)],
            "steps": [{"agent": name, "input": {"source": "initial"}, "depends_on": []}],
        }
    )


def _relaxed(old: dict, new: dict) -> bool:
    return not schema_relaxation_diff(_cfg(old), _cfg(new)).ok


# ---------------------------------------------------------------------------
# Decidable relaxations: MUST flag
# ---------------------------------------------------------------------------

def test_maxlength_increase_flags():
    assert _relaxed(
        {"type": "object", "properties": {"x": {"type": "string", "maxLength": 5}}},
        {"type": "object", "properties": {"x": {"type": "string", "maxLength": 10}}},
    )


def test_minlength_decrease_flags():
    assert _relaxed(
        {"type": "object", "properties": {"x": {"type": "string", "minLength": 5}}},
        {"type": "object", "properties": {"x": {"type": "string", "minLength": 2}}},
    )


def test_maxitems_increase_flags():
    assert _relaxed(
        {"type": "object", "properties": {"x": {"type": "array", "maxItems": 2}}},
        {"type": "object", "properties": {"x": {"type": "array", "maxItems": 9}}},
    )


def test_minitems_decrease_flags():
    assert _relaxed(
        {"type": "object", "properties": {"x": {"type": "array", "minItems": 3}}},
        {"type": "object", "properties": {"x": {"type": "array", "minItems": 1}}},
    )


def test_minimum_decrease_flags():
    assert _relaxed(
        {"type": "object", "properties": {"x": {"type": "number", "minimum": 0}}},
        {"type": "object", "properties": {"x": {"type": "number", "minimum": -10}}},
    )


def test_maximum_increase_flags():
    assert _relaxed(
        {"type": "object", "properties": {"x": {"type": "number", "maximum": 100}}},
        {"type": "object", "properties": {"x": {"type": "number", "maximum": 500}}},
    )


def test_exclusive_minimum_decrease_flags():
    assert _relaxed(
        {"type": "object", "properties": {"x": {"type": "number", "exclusiveMinimum": 0}}},
        {"type": "object", "properties": {"x": {"type": "number", "exclusiveMinimum": -5}}},
    )


def test_numeric_bound_dropped_flags():
    # maximum present in old, absent in new -> constraint dropped (category 3)
    assert _relaxed(
        {"type": "object", "properties": {"x": {"type": "number", "maximum": 100}}},
        {"type": "object", "properties": {"x": {"type": "number"}}},
    )


def test_enum_superset_flags():
    assert _relaxed(
        {"type": "object", "properties": {"x": {"enum": ["a", "b"]}}},
        {"type": "object", "properties": {"x": {"enum": ["a", "b", "c"]}}},
    )


def test_enum_disjoint_flags():
    # [a,b] -> [b,c]: c is newly accepted
    assert _relaxed(
        {"type": "object", "properties": {"x": {"enum": ["a", "b"]}}},
        {"type": "object", "properties": {"x": {"enum": ["b", "c"]}}},
    )


def test_const_change_flags():
    assert _relaxed(
        {"type": "object", "properties": {"x": {"const": 42}}},
        {"type": "object", "properties": {"x": {"const": 43}}},
    )


def test_multipleof_widens_flags():
    # old multiples of 6, new multiples of 3 -> wider (new divides old)
    assert _relaxed(
        {"type": "object", "properties": {"x": {"type": "number", "multipleOf": 6}}},
        {"type": "object", "properties": {"x": {"type": "number", "multipleOf": 3}}},
    )


def test_multipleof_neither_divides_flags():
    # 6 -> 4: neither divides the other
    assert _relaxed(
        {"type": "object", "properties": {"x": {"type": "number", "multipleOf": 6}}},
        {"type": "object", "properties": {"x": {"type": "number", "multipleOf": 4}}},
    )


def test_required_entry_removed_flags():
    assert _relaxed(
        {"type": "object", "required": ["x", "y"], "properties": {"x": {}, "y": {}}},
        {"type": "object", "required": ["x"], "properties": {"x": {}, "y": {}}},
    )


def test_required_keyword_removed_flags():
    assert _relaxed(
        {"type": "object", "required": ["x"], "properties": {"x": {}}},
        {"type": "object", "properties": {"x": {}}},
    )


def test_literal_to_str_flags():
    # enum dropped, type unchanged (the Literal[...] -> str refactor; category 3)
    assert _relaxed(
        {"type": "object", "properties": {"m": {"type": "string", "enum": ["a", "b"]}}},
        {"type": "object", "properties": {"m": {"type": "string"}}},
    )


def test_pattern_dropped_flags():
    assert _relaxed(
        {"type": "object", "properties": {"x": {"type": "string", "pattern": "^[0-9]+$"}}},
        {"type": "object", "properties": {"x": {"type": "string"}}},
    )


def test_pattern_changed_flags():
    # regex containment is undecidable -> any change fails closed
    assert _relaxed(
        {"type": "object", "properties": {"x": {"type": "string", "pattern": "^[0-9]+$"}}},
        {"type": "object", "properties": {"x": {"type": "string", "pattern": "^[0-9]*$"}}},
    )


def test_type_removed_flags():
    assert _relaxed(
        {"type": "object", "properties": {"x": {"type": "string"}}},
        {"type": "object", "properties": {"x": {}}},
    )


def test_type_changed_flags():
    assert _relaxed(
        {"type": "object", "properties": {"x": {"type": "integer"}}},
        {"type": "object", "properties": {"x": {"type": "number"}}},
    )


def test_anyof_changed_flags():
    assert _relaxed(
        {"type": "object", "properties": {"x": {"anyOf": [{"type": "string"}]}}},
        {"type": "object", "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}}},
    )


def test_additionalprops_false_to_true_flags():
    assert _relaxed(
        {"type": "object", "additionalProperties": False, "properties": {"a": {"type": "string"}}},
        {"type": "object", "additionalProperties": True, "properties": {"a": {"type": "string"}}},
    )


def test_additionalprops_removed_flags():
    # old forbade extras, new (absent) permits them
    assert _relaxed(
        {"type": "object", "additionalProperties": False, "properties": {"a": {"type": "string"}}},
        {"type": "object", "properties": {"a": {"type": "string"}}},
    )


def test_new_optional_field_under_forbid_flags():
    assert _relaxed(
        {"type": "object", "additionalProperties": False, "properties": {"a": {"type": "string"}}},
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        },
    )


def test_nested_defs_relaxation_flags():
    old = {
        "type": "object",
        "properties": {"addr": {"$ref": "#/$defs/A"}},
        "$defs": {"A": {"type": "object", "properties": {"zip": {"type": "string", "maxLength": 5}}}},
    }
    new = {
        "type": "object",
        "properties": {"addr": {"$ref": "#/$defs/A"}},
        "$defs": {"A": {"type": "object", "properties": {"zip": {"type": "string", "maxLength": 9}}}},
    }
    assert _relaxed(old, new)


def test_nested_defs_relaxation_path_is_traversal_route():
    old = {
        "type": "object",
        "properties": {"addr": {"$ref": "#/$defs/A"}},
        "$defs": {"A": {"type": "object", "properties": {"zip": {"type": "string", "maxLength": 5}}}},
    }
    new = {
        "type": "object",
        "properties": {"addr": {"$ref": "#/$defs/A"}},
        "$defs": {"A": {"type": "object", "properties": {"zip": {"type": "string", "maxLength": 9}}}},
    }
    diff = schema_relaxation_diff(_cfg(old), _cfg(new))
    assert not diff.ok
    paths = {r.path for r in diff.relaxations}
    assert "/$defs/A/properties/zip/maxLength" in paths


def test_ref_target_change_flags():
    old = {
        "type": "object",
        "properties": {"addr": {"$ref": "#/$defs/A"}},
        "$defs": {"A": {"type": "object"}, "B": {"type": "object"}},
    }
    new = {
        "type": "object",
        "properties": {"addr": {"$ref": "#/$defs/B"}},
        "$defs": {"A": {"type": "object"}, "B": {"type": "object"}},
    }
    assert _relaxed(old, new)


def test_recursive_defs_terminates():
    # a self-referential $defs entry must not infinite-loop
    schema = {
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"next": {"$ref": "#/$defs/Node"}, "v": {"type": "string", "maxLength": 5}},
            }
        },
    }
    relaxed = {
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"next": {"$ref": "#/$defs/Node"}, "v": {"type": "string", "maxLength": 50}},
            }
        },
    }
    assert not _relaxed(schema, schema)  # identity
    assert _relaxed(schema, relaxed)  # nested maxLength raised inside recursive model


# ---------------------------------------------------------------------------
# Tightenings / identity / safe additions: MUST pass
# ---------------------------------------------------------------------------

def test_identity_passes():
    s = {"type": "object", "properties": {"x": {"type": "string", "maxLength": 5}}}
    assert not _relaxed(s, dict(s))


def test_maxlength_decrease_passes():
    assert not _relaxed(
        {"type": "object", "properties": {"x": {"type": "string", "maxLength": 10}}},
        {"type": "object", "properties": {"x": {"type": "string", "maxLength": 5}}},
    )


def test_minimum_increase_passes():
    assert not _relaxed(
        {"type": "object", "properties": {"x": {"type": "number", "minimum": 0}}},
        {"type": "object", "properties": {"x": {"type": "number", "minimum": 10}}},
    )


def test_enum_subset_passes():
    assert not _relaxed(
        {"type": "object", "properties": {"x": {"enum": ["a", "b", "c"]}}},
        {"type": "object", "properties": {"x": {"enum": ["a", "b"]}}},
    )


def test_multipleof_narrows_passes():
    # old multiples of 3, new multiples of 6 -> narrower (old divides new)
    assert not _relaxed(
        {"type": "object", "properties": {"x": {"type": "number", "multipleOf": 3}}},
        {"type": "object", "properties": {"x": {"type": "number", "multipleOf": 6}}},
    )


def test_new_constraint_added_passes():
    assert not _relaxed(
        {"type": "object", "properties": {"x": {"type": "string"}}},
        {"type": "object", "properties": {"x": {"type": "string", "maxLength": 5}}},
    )


def test_required_entry_added_passes():
    assert not _relaxed(
        {"type": "object", "required": ["x"], "properties": {"x": {}, "y": {}}},
        {"type": "object", "required": ["x", "y"], "properties": {"x": {}, "y": {}}},
    )


def test_additionalprops_true_to_false_passes():
    assert not _relaxed(
        {"type": "object", "additionalProperties": True, "properties": {"a": {"type": "string"}}},
        {"type": "object", "additionalProperties": False, "properties": {"a": {"type": "string"}}},
    )


def test_metadata_removed_ignored():
    assert not _relaxed(
        {"type": "object", "title": "T", "properties": {"x": {"type": "string", "description": "d"}}},
        {"type": "object", "properties": {"x": {"type": "string"}}},
    )


def test_metadata_changed_ignored():
    assert not _relaxed(
        {"type": "object", "title": "Old", "properties": {"x": {"type": "string", "default": "a"}}},
        {"type": "object", "title": "New", "properties": {"x": {"type": "string", "default": "b"}}},
    )


def test_new_optional_field_without_forbid_passes():
    assert not _relaxed(
        {"type": "object", "properties": {"a": {"type": "string"}}},
        {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}},
    )


def test_property_removed_under_forbid_passes():
    # removing a field under additionalProperties:false is provably a narrowing
    assert not _relaxed(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        },
        {"type": "object", "additionalProperties": False, "properties": {"a": {"type": "string"}}},
    )


def test_property_removed_under_permissive_flags():
    # under permissive additionalProperties, removing a field drops its type
    # constraint while the field can still appear -> wider -> flag
    assert _relaxed(
        {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}},
        {"type": "object", "properties": {"a": {"type": "string"}}},
    )


# ---------------------------------------------------------------------------
# Multi-agent + legibility
# ---------------------------------------------------------------------------

def _multi(schemas: dict[str, dict]) -> PipelineConfig:
    agents = [_agent(name, s) for name, s in schemas.items()]
    steps = [{"agent": n, "input": {"source": "initial"}, "depends_on": []} for n in schemas]
    return PipelineConfig.model_validate(
        {
            "schema_version": 1,
            "pipeline": {"name": "p", "version": "1.0"},
            "agents": agents,
            "steps": steps,
        }
    )


def test_relaxation_in_one_of_many_agents_caught():
    old = _multi(
        {
            "a": {"type": "object", "properties": {"x": {"type": "string", "maxLength": 5}}},
            "b": {"type": "object", "properties": {"y": {"type": "string", "maxLength": 5}}},
        }
    )
    new = _multi(
        {
            "a": {"type": "object", "properties": {"x": {"type": "string", "maxLength": 5}}},
            "b": {"type": "object", "properties": {"y": {"type": "string", "maxLength": 50}}},
        }
    )
    diff = schema_relaxation_diff(old, new)
    assert not diff.ok
    assert {r.agent for r in diff.relaxations} == {"b"}


def test_added_agent_ignored():
    old = _multi({"a": {"type": "object"}})
    new = _multi({"a": {"type": "object"}, "b": {"type": "object", "properties": {}}})
    assert schema_relaxation_diff(old, new).ok


def test_removed_agent_ignored():
    old = _multi({"a": {"type": "object"}, "b": {"type": "object"}})
    new = _multi({"a": {"type": "object"}})
    assert schema_relaxation_diff(old, new).ok


def test_legible_names_agent_path_keyword():
    old = {"type": "object", "properties": {"x": {"type": "string", "maxLength": 5}}}
    new = {"type": "object", "properties": {"x": {"type": "string", "maxLength": 10}}}
    diff = schema_relaxation_diff(_cfg(old, name="loader"), _cfg(new, name="loader"))
    text = diff.legible()
    assert "loader" in text
    assert "/properties/x/maxLength" in text
    assert "maxLength" in text
    assert diff.certificate() == diff.legible()


def test_diff_fingerprints_cover_input_schemas():
    old = _cfg({"type": "object", "properties": {"x": {"type": "string", "maxLength": 5}}})
    new = _cfg({"type": "object", "properties": {"x": {"type": "string", "maxLength": 10}}})
    diff = schema_relaxation_diff(old, new)
    assert diff.old_fingerprint.startswith("sha256:")
    assert diff.new_fingerprint.startswith("sha256:")
    assert diff.old_fingerprint != diff.new_fingerprint


def test_diff_is_dataclass_shapes():
    diff = schema_relaxation_diff(_cfg({"type": "object"}), _cfg({"type": "object"}))
    assert isinstance(diff, SchemaRelaxationDiff)
    assert isinstance(diff.relaxations, tuple)
    assert diff.ok


# ---------------------------------------------------------------------------
# check_no_schema_relaxation wrapper
# ---------------------------------------------------------------------------

def test_check_raises_on_relaxation():
    with pytest.raises(SchemaRelaxationError):
        check_no_schema_relaxation(
            _cfg({"type": "object", "properties": {"x": {"type": "string", "maxLength": 5}}}),
            _cfg({"type": "object", "properties": {"x": {"type": "string", "maxLength": 10}}}),
        )


def test_check_error_carries_diff():
    try:
        check_no_schema_relaxation(
            _cfg({"type": "object", "properties": {"x": {"type": "string", "maxLength": 5}}}),
            _cfg({"type": "object", "properties": {"x": {"type": "string", "maxLength": 10}}}),
        )
    except SchemaRelaxationError as exc:
        assert getattr(exc, "diff", None) is not None
        assert not exc.diff.ok
    else:
        raise AssertionError("expected SchemaRelaxationError")


def test_check_passes_on_tightening():
    result = check_no_schema_relaxation(
        _cfg({"type": "object", "properties": {"x": {"type": "string", "maxLength": 10}}}),
        _cfg({"type": "object", "properties": {"x": {"type": "string", "maxLength": 5}}}),
    )
    assert result is None


def test_check_passes_on_identity():
    s = {"type": "object", "properties": {"x": {"type": "string"}}}
    assert check_no_schema_relaxation(_cfg(s), _cfg(dict(s))) is None


def test_schema_relaxation_is_frozen_dataclass():
    r = SchemaRelaxation(agent="a", path="/properties/x/maxLength", keyword="maxLength", old=5, new=10, reason="r")
    with pytest.raises(Exception):
        r.agent = "b"  # type: ignore[misc]
