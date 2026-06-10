"""Context isolation — orchestrator-constructed agent inputs.

The orchestrator builds each agent's input payload programmatically from declared
bindings, passing only the fields the agent's input schema requires. Combined with
schema validation (which drops any extra fields), an agent is physically incapable
of seeing fields it was not given.
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel


def build_input(
    inputs: Mapping[str, Any],
    outputs: Mapping[str, BaseModel],
    *,
    initial: BaseModel | None = None,
    predecessor: str | None = None,
) -> Any:
    """Construct an agent's input payload.

    - With explicit ``inputs`` bindings: a dict of exactly the bound fields, each
      pulled from the named upstream output (whole object or one field).
    - Without bindings: the pipeline ``initial`` for the first step (``predecessor``
      is None), otherwise the immediate predecessor's whole output.

    ``inputs`` values are duck-typed bindings exposing ``.agent`` and ``.field``
    (see ``drawbore.pipeline.From``); this module does not import the pipeline.

    Callers pass bindings already validated at pipeline registration (the source
    agent exists and precedes this step) and outputs that are validated model
    instances, so missing-key/attribute errors here indicate a framework bug and
    are allowed to surface.
    """
    if not inputs:
        if predecessor is None:
            return initial
        return outputs[predecessor].model_dump()
    payload: dict[str, Any] = {}
    for target_field, src in inputs.items():
        source_obj = outputs[src.agent]
        payload[target_field] = (
            source_obj.model_dump() if src.field is None
            else getattr(source_obj, src.field)
        )
    return payload
