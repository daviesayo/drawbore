"""Schema enforcement errors."""

from __future__ import annotations

from typing import Any


def render_schema_errors(errors: list[dict[str, Any]]) -> str:
    """Render Pydantic error dicts as a regulator-readable summary
    (``"field: message; ..."``), never a raw dict dump with error URLs. Shared by
    the pipeline executor (halt reasons) and the tool loop (the field errors that
    seed a single corrective schema reprompt)."""
    parts = []
    for e in errors:
        loc = ".".join(str(p) for p in e.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {e.get('msg', 'invalid')}")
    return "; ".join(parts) if parts else "validation failed"


class SchemaError(Exception):
    """Base class for all schema enforcement failures."""


class SchemaCompatibilityError(SchemaError):
    """Raised at pipeline registration when a binding's source type cannot
    satisfy the declared target type (static check)."""


class SchemaValidationError(SchemaError):
    """Raised at a runtime agent boundary when a value does not conform to its
    declared Pydantic model."""

    def __init__(self, model: type, errors: list[dict[str, Any]]):
        self.model = model
        self.errors = errors
        super().__init__(
            f"{getattr(model, '__name__', model)} validation failed: {errors}"
        )
