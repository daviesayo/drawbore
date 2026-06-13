"""Structural guard: the drift check must cover every serializable agent field.

``_check_declaration`` derives the expected declaration from ``_agent_config``
and loops over ``AgentConfig.model_fields``, so it covers any new field
automatically. This test guards the other half of the contract: that the
``AgentSpec`` field set and the ``AgentConfig`` field set never silently
diverge. If they do (a new spec field with no config field, or vice versa),
this fails loudly instead of skipping a drift check at runtime.
"""

import dataclasses

from drawbore.agent.spec import AgentSpec
from drawbore.config.models import AgentConfig

# AgentSpec fields NOT serialized 1:1 into AgentConfig:
_SPEC_NOT_DIRECT = {
    "fn",  # runtime callable, never serialized
    "input",  # expands to input_schema + input_schema_hash
    "output",  # expands to output_schema + output_schema_hash
}
# AgentConfig fields with no same-named AgentSpec field:
_CONFIG_NOT_FROM_SPEC = {
    "ref",  # catalog key, not a spec value
    "input_schema",  # derived from spec.input
    "output_schema",  # derived from spec.output
    "input_schema_hash",  # derived from spec.input
    "output_schema_hash",  # derived from spec.output
}


def test_agent_config_covers_every_serializable_spec_field():
    spec_fields = {f.name for f in dataclasses.fields(AgentSpec)} - _SPEC_NOT_DIRECT
    config_fields = set(AgentConfig.model_fields) - _CONFIG_NOT_FROM_SPEC
    assert spec_fields == config_fields, (
        "AgentSpec and AgentConfig field sets diverged. Update AgentConfig, "
        "_agent_config, and the documented exclusions in this test together.\n"
        f"  in spec, missing from config: {spec_fields - config_fields}\n"
        f"  in config, not a spec field:  {config_fields - spec_fields}"
    )
