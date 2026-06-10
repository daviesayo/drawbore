"""Schema enforcement errors."""

from __future__ import annotations

from typing import Any


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
