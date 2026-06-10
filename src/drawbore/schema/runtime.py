"""Runtime schema validation at agent boundaries."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import SchemaValidationError

M = TypeVar("M", bound=BaseModel)


def validate(model: type[M], value: Any) -> M:
    """Validate ``value`` against ``model`` in strict mode.

    Returns a validated instance of ``model``. Raises
    :class:`SchemaValidationError` on any nonconformance. Strict mode means no
    lax type coercion — a wrong type fails rather than being silently converted.
    """
    try:
        return model.model_validate(value, strict=True)
    except ValidationError as exc:
        raise SchemaValidationError(model, exc.errors()) from exc
