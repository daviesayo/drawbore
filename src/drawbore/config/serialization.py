"""Pipeline -> manifest export.

Walk a live ``Pipeline`` to a ``PipelineConfig``. Derive each step's only valid
tagged input mode from runtime semantics and FAIL CLOSED if a live ``Step``'s
``depends_on`` does not match the derived mode — export must not canonicalize a
lying ``depends_on``, because that would hide live/manifest disagreement."""

from __future__ import annotations

import json
from typing import Mapping

from drawbore.agent import Agent
from drawbore.escalation import EscalationPolicy
from drawbore.evidence import EvidencePolicy
from drawbore.pipeline import Pipeline
from drawbore.pipeline.graph import JoinNode
from drawbore.pipeline.pipeline import Step

from .catalog import AgentCatalog, ref_for
from .errors import ConfigResolutionError
from .fingerprint import schema_fingerprint
from .models import (
    AgentConfig,
    BindingConfig,
    ConditionConfig,
    EvidencePolicyConfig,
    JoinConfig,
    OnFailureConfig,
    PipelineConfig,
    PipelineMetaConfig,
    StepConfig,
    StepInputConfig,
)

_SCHEMA_VERSION = 1


def to_config(
    pipeline: Pipeline, *, agents: "AgentCatalog | Mapping[str, Agent]"
) -> PipelineConfig:
    """Emit a typed manifest for ``pipeline``. Fails closed if any agent is
    uncataloged/ambiguous or any step's ``depends_on`` lies."""
    on_failure = _on_failure_config(pipeline._on_failure)
    meta = PipelineMetaConfig(
        name=pipeline.name,
        version=pipeline.version,
        confidence_threshold=pipeline._confidence_threshold,
        on_failure=on_failure,
    )

    agent_configs = [
        _agent_config(node.agent, ref_for(agents, node.agent))
        for node in pipeline.steps
        if not isinstance(node, JoinNode)
    ]
    step_configs = [
        _node_config(idx, node, pipeline.steps) for idx, node in enumerate(pipeline.steps)
    ]

    return PipelineConfig(
        schema_version=_SCHEMA_VERSION, pipeline=meta,
        agents=agent_configs, steps=step_configs,
    )


def to_json(pipeline: Pipeline, *, agents: "AgentCatalog | Mapping[str, Agent]") -> str:
    """Deterministic canonical JSON of the manifest (sorted keys, compact)."""
    config = to_config(pipeline, agents=agents)
    return json.dumps(
        config.model_dump(mode="json", by_alias=True),
        sort_keys=True, separators=(",", ":"),
    )


def _on_failure_config(policy: EscalationPolicy | None) -> OnFailureConfig | None:
    if policy is None:
        return None
    return OnFailureConfig(channel=policy.channel, target=policy.target, mode=policy.mode)


def _evidence_config(policy: EvidencePolicy | None) -> EvidencePolicyConfig | None:
    if policy is None:
        return None
    # EvidencePolicy is a pydantic model; mirror its fields into the strict config
    # model (policy only — never stores/handles).
    return EvidencePolicyConfig(**policy.model_dump())


def _agent_config(agent: Agent, ref: str) -> AgentConfig:
    spec = agent.spec
    return AgentConfig(
        name=spec.name,
        ref=ref,
        version=spec.version,
        risk_tier=spec.risk_tier,
        requires_human_approval=spec.requires_human_approval,
        context_access=spec.context_access,
        tools=list(spec.tools),
        model=spec.model,
        fallback_model=spec.fallback_model,
        instructions=spec.instructions,
        input_schema=spec.input.model_json_schema(),
        output_schema=spec.output.model_json_schema(),
        input_schema_hash=schema_fingerprint(spec.input),
        output_schema_hash=schema_fingerprint(spec.output),
    )


def _node_config(idx: int, node: "Step | JoinNode", nodes: list["Step | JoinNode"]):
    """Emit a ``JoinConfig`` for a join node, else delegate to the agent ``_step_config``."""
    if isinstance(node, JoinNode):
        bindings = (
            {f: BindingConfig.model_validate({"from": src.ref}) for f, src in node.inputs.items()}
            if node.inputs
            else None
        )
        return JoinConfig(
            name=node.name,
            sources=list(node.sources),
            policy=node.policy,
            output_schema=node.output.model_json_schema(),
            output_schema_hash=schema_fingerprint(node.output),
            bindings=bindings,
        )
    return _step_config(idx, node, nodes)


def _step_config(idx: int, step: Step, steps: list["Step | JoinNode"]) -> StepConfig:
    input_mode, expected_depends_on = _derive_input(idx, step, steps)
    if sorted(step.depends_on) != sorted(expected_depends_on):
        raise _depends_on_error(idx, step, expected_depends_on)
    return StepConfig(
        agent=step.agent.name,
        input=input_mode,
        depends_on=list(expected_depends_on),
        when=_condition_config(step.when),
        evidence=_evidence_config(step.evidence),
    )


def _condition_config(when) -> ConditionConfig | None:
    if when is None:
        return None
    return ConditionConfig(
        ref=when.ref,
        equals=when.equals,
        in_=when.in_,
        is_true=when.is_true,
        gt=when.gt,
        lt=when.lt,
    )


def _derive_input(
    idx: int, step: Step, steps: list["Step | JoinNode"]
) -> tuple[StepInputConfig, list[str]]:
    """Return ``(tagged_input, expected_depends_on)`` for a live step."""
    if step.inputs:
        bindings = {
            field: BindingConfig.model_validate({"from": src.ref})
            for field, src in step.inputs.items()
        }
        expected = sorted({src.agent for src in step.inputs.values()})
        return StepInputConfig(bindings=bindings), expected
    if idx == 0:
        return StepInputConfig(source="initial"), []
    predecessor = (
        steps[idx - 1].name
        if isinstance(steps[idx - 1], JoinNode)
        else steps[idx - 1].agent.name
    )
    return StepInputConfig(source="previous", agent=predecessor), [predecessor]


def _depends_on_error(idx: int, step: Step, expected: list[str]) -> ConfigResolutionError:
    if step.inputs:
        return ConfigResolutionError(
            f"step {idx} ('{step.agent.name}') depends_on {step.depends_on} does not "
            f"match its binding sources {expected}"
        )
    if idx == 0:
        return ConfigResolutionError(
            f"step 0 ('{step.agent.name}') source='initial' requires depends_on=[], "
            f"got {step.depends_on}"
        )
    return ConfigResolutionError(
        f"step {idx} ('{step.agent.name}') source='previous' must name the immediate "
        f"predecessor {expected}, got depends_on={step.depends_on}"
    )
