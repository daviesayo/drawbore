"""Breaking vs non-breaking change classification.

The framework distinguishes at registration time:

- Non-breaking: adding an OPTIONAL field, updating a prompt, swapping the model.
- Breaking: modifying a REQUIRED field, removing a field, changing a field type.

This classifier compares two agent versions' input AND output models using
Pydantic field introspection (field annotations are the type oracle).
Field types are compared via a STRUCTURAL key (``TypeAdapter(annotation).json_schema()``)
rather than ``str(annotation)``, so an in-place change to a nested model (which keeps
the same class name) is still detected. Prompt/model swaps are not schema-visible, so
two versions with identical schemas classify as non-breaking.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, TypeAdapter

ChangeKind = Literal["breaking", "non_breaking"]


def _type_key(annotation) -> str:
    """A structural key for a field's type: its JSON schema, which recursively
    expands nested models and normalises generic-alias spellings. Used as a
    change-detection sentinel (not a type-assignability oracle). Falls back to
    ``str`` for annotations Pydantic cannot schematise."""
    try:
        return json.dumps(TypeAdapter(annotation).json_schema(), sort_keys=True)
    except Exception:
        return str(annotation)


def _diff_is_breaking(old: type[BaseModel], new: type[BaseModel]) -> bool:
    old_fields = old.model_fields
    new_fields = new.model_fields

    for name, old_info in old_fields.items():
        if name not in new_fields:
            return True  # removed field
        new_info = new_fields[name]
        if _type_key(old_info.annotation) != _type_key(new_info.annotation):
            return True  # changed field type (structural — catches nested changes)
        if old_info.is_required() != new_info.is_required():
            return True  # modified a field's required-ness

    for name, new_info in new_fields.items():
        if name not in old_fields and new_info.is_required():
            return True  # added a required field

    return False


def classify_change(
    old_input: type[BaseModel],
    old_output: type[BaseModel],
    new_input: type[BaseModel],
    new_output: type[BaseModel],
) -> ChangeKind:
    """Classify the change between two agent versions. Returns
    ``"breaking"`` if either the input or output schema has a breaking change,
    else ``"non_breaking"`` (identical schemas included)."""
    if _diff_is_breaking(old_input, new_input):
        return "breaking"
    if _diff_is_breaking(old_output, new_output):
        return "breaking"
    return "non_breaking"
