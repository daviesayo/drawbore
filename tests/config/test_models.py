import json

import pytest
from pydantic import ValidationError

from drawbore.config.models import (
    AgentConfig,
    BindingConfig,
    EvidencePolicyConfig,
    OnFailureConfig,
    PipelineConfig,
    PipelineMetaConfig,
    StepConfig,
    StepInputConfig,
)


def _agent(**over):
    base = dict(
        name="screen", ref="pkg:screen", version="1.0.0", risk_tier="high",
        requires_human_approval=False, context_access="none", tools=["t.read"],
        model=None, fallback_model=None, instructions=None,
        input_schema={}, output_schema={},
        input_schema_hash="sha256:a", output_schema_hash="sha256:b",
    )
    base.update(over)
    return base


def test_extra_forbidden_at_top_level():
    with pytest.raises(ValidationError):
        PipelineConfig(
            schema_version=1,
            pipeline=PipelineMetaConfig(name="p", version="0.1.0"),
            agents=[], steps=[], surprise=1,
        )


def test_extra_forbidden_on_pipeline_meta():
    with pytest.raises(ValidationError):
        PipelineMetaConfig(name="p", version="0.1.0", surprise=1)


def test_extra_forbidden_on_on_failure():
    with pytest.raises(ValidationError):
        OnFailureConfig(channel="c", target="t", mode="sync", surprise=1)


def test_extra_forbidden_on_agent():
    with pytest.raises(ValidationError):
        AgentConfig(**_agent(surprise=1))


def test_extra_forbidden_on_step():
    with pytest.raises(ValidationError):
        StepConfig(agent="a", input=StepInputConfig(source="initial"), surprise=1)


def test_extra_forbidden_on_step_input():
    with pytest.raises(ValidationError):
        StepInputConfig(source="initial", surprise=1)


def test_extra_forbidden_on_binding():
    with pytest.raises(ValidationError):
        BindingConfig.model_validate({"from": "a.b", "surprise": 1})


def test_extra_forbidden_on_evidence_policy():
    with pytest.raises(ValidationError):
        EvidencePolicyConfig(name="e", enabled=True, surprise=1)


def test_binding_uses_from_alias():
    b = BindingConfig.model_validate({"from": "fetch.transaction"})
    assert b.from_ == "fetch.transaction"
    assert b.model_dump(by_alias=True) == {"from": "fetch.transaction"}


def test_binding_rejects_literal_with_a_legible_message():
    # Literal bindings are not supported — reject with a clear, literal-naming message.
    with pytest.raises(ValidationError, match="literal input bindings are not supported"):
        BindingConfig.model_validate({"literal": 42})


def test_step_input_initial_is_well_formed():
    s = StepInputConfig(source="initial")
    assert s.source == "initial" and s.agent is None and s.bindings is None


def test_step_input_previous_requires_agent():
    StepInputConfig(source="previous", agent="fetch")          # ok
    with pytest.raises(ValidationError):
        StepInputConfig(source="previous")                     # missing agent


def test_step_input_bindings_cannot_mix_with_source():
    with pytest.raises(ValidationError):
        StepInputConfig(source="initial", bindings={"x": BindingConfig.model_validate({"from": "a.b"})})


def test_step_input_initial_rejects_agent():
    # 'initial' is the first step fed the pipeline input — it names no predecessor.
    with pytest.raises(ValidationError):
        StepInputConfig(source="initial", agent="fetch")


def test_step_input_bindings_cannot_mix_with_agent():
    with pytest.raises(ValidationError):
        StepInputConfig(agent="fetch", bindings={"x": BindingConfig.model_validate({"from": "a.b"})})


def test_step_input_requires_a_mode():
    with pytest.raises(ValidationError):
        StepInputConfig()                                      # neither source nor bindings


def test_evidence_policy_config_dump_serialization_contract():
    # Guards the serializer (Task 5): plain model_dump() returns a tuple for
    # allowed_transforms (lossless into EvidencePolicy(**...)), but JSON output MUST
    # go through mode="json", which renders it as a list. Documents the asymmetry.
    cfg = EvidencePolicyConfig()
    assert isinstance(cfg.model_dump()["allowed_transforms"], tuple)
    dumped = cfg.model_dump(mode="json")
    assert isinstance(dumped["allowed_transforms"], list)
    json.dumps(dumped)                                          # must not raise


def test_binding_plain_dump_uses_python_name_alias_needs_by_alias():
    # Guards the serializer (Task 5): the JSON key is 'from', but a plain dump emits
    # the Python field name 'from_'. Serialization MUST pass by_alias=True.
    b = BindingConfig.model_validate({"from": "a.b"})
    assert "from_" in b.model_dump() and "from" not in b.model_dump()
    assert b.model_dump(by_alias=True) == {"from": "a.b"}
