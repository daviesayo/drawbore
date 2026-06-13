"""Centralised model-input construction.

``build_model_request`` is the ONE place a validated step payload becomes model
input. Keeping it centralised means that a future pre-model transform hooks HERE —
it will inspect the payload + an evidence policy and may return a compressed view
plus audit decisions. This function implements identity passthrough only: no
transform, no evidence policy, no compression. Do not add any of that here.
"""

from __future__ import annotations

import json

from .request import ModelRequest


def build_model_request(
    spec, payload, *, model_chain: tuple[str, ...], native: bool = False
) -> ModelRequest:
    """Assemble the non-streaming model request for a model-backed agent.

    The system message carries the agent's instructions and the JSON output
    contract (the output model's JSON schema); the user message is the validated
    input rendered as JSON data — never as instructions (input is data, not
    authority). ``spec`` is an ``AgentSpec``; ``payload`` is the
    validated input ``BaseModel`` instance. ``spec``/``payload`` are intentionally
    un-annotated to keep ``llm`` from importing ``agent``.

    Extension seam: a future pre-model transform will wrap this assembly to return
    ``(ModelRequest, decisions)`` for evidence compression. Keep this the single
    assembly point so that change is local.
    """
    output_schema = json.dumps(spec.output.model_json_schema(), sort_keys=True)
    instructions = (spec.instructions or "").strip()
    system = (
        (instructions + "\n\n" if instructions else "")
        + "Respond ONLY with a single JSON object that matches this schema, with "
        "no prose and no code fences:\n"
        + output_schema
    )
    user = json.dumps(payload.model_dump(), sort_keys=True)
    # When ``native`` is set, carry the output class so the gateway can request
    # provider-constrained decoding. The system-prompt schema instruction above is
    # UNCHANGED in both cases (belt-and-suspenders): ``system`` is byte-identical
    # whether or not native mode is on.
    output_format = spec.output if native else None
    return ModelRequest(
        system=system, user=user, model_chain=model_chain, output_format=output_format
    )
