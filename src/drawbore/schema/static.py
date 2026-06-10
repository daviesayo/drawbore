"""Static schema compatibility — structural width subtyping.

Compatibility is computed from Python type annotations (Pydantic ``model_fields``
expose them), NOT from JSON Schema, which is too lossy to be the type oracle.
``.model_json_schema()`` is reserved for the config layer, docs, and audit
evidence elsewhere.
"""

from __future__ import annotations

from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

from .errors import SchemaCompatibilityError


def _is_optional(annotation: Any) -> bool:
    return get_origin(annotation) is Union and type(None) in get_args(annotation)


def _strip_optional(annotation: Any) -> Any:
    if get_origin(annotation) is Union:
        args = tuple(a for a in get_args(annotation) if a is not type(None))
        return args[0] if len(args) == 1 else Union[args]
    return annotation


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _model_assignable(source: type, target: type) -> bool:
    """Structural width subtyping between two Pydantic models:
    ``source`` satisfies ``target`` if it supplies every required target field
    with an assignable type. Extra source fields are ignored; optional target
    fields may be absent.
    """
    src_fields = source.model_fields
    for name, finfo in target.model_fields.items():
        if name in src_fields:
            if not is_assignable(src_fields[name].annotation, finfo.annotation):
                return False
        elif finfo.is_required():
            return False
    return True


def is_assignable(source: Any, target: Any) -> bool:
    """True if a value typed ``source`` can satisfy a field typed ``target``.

    Rules (structural width subtyping):
    - ``target is Any`` accepts anything;
    - identical annotations are assignable;
    - a non-optional ``T`` satisfies ``Optional[T]``, but ``Optional[T]`` does
      not satisfy a non-optional ``T``;
    - two Pydantic models compare structurally (every required target field is
      present in the source with an assignable type; extra source fields ignored);
    - for other plain classes, ``source`` is assignable to ``target`` when
      ``issubclass(source, target)``.
    """
    if target is Any:
        return True
    if source == target:
        return True
    if _is_optional(target):
        inner_source = _strip_optional(source) if _is_optional(source) else source
        return is_assignable(inner_source, _strip_optional(target))
    if _is_optional(source):
        return False
    if _is_model(source) and _is_model(target):
        return _model_assignable(source, target)
    if isinstance(source, type) and isinstance(target, type):
        return issubclass(source, target)
    return False


def check_compatibility(source_annotation: Any, target_annotation: Any) -> bool:
    """Return ``True`` if ``source_annotation`` satisfies ``target_annotation``,
    else raise :class:`SchemaCompatibilityError`. Used at pipeline registration.
    """
    if is_assignable(source_annotation, target_annotation):
        return True
    raise SchemaCompatibilityError(
        f"source type {source_annotation!r} cannot satisfy "
        f"target type {target_annotation!r}"
    )
