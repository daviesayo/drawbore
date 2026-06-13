"""Manifest -> pipeline import.

Fail-closed: parse, reject unknown schema_version, resolve refs, verify every
resolved declaration matches the config (drift fails closed), reject duplicate
names / lying input modes / literal bindings, verify tools are registered,
then rebuild via the existing ``Pipeline``/``Pipeline.add``. ``model + tools +
fallback_model`` is accepted at import; execution enforces the safe loop fallback
rules. Any lower-level failure surfaces as ``ConfigResolutionError`` with the
reason text preserved."""

from __future__ import annotations

import json
from typing import Any, Mapping

from pydantic import ValidationError

from drawbore.agent import Agent
from drawbore.escalation import EscalationPolicy
from drawbore.evidence import EvidencePolicy
from drawbore.pipeline import Join, Pipeline, When
from drawbore.pipeline.binding import From
from drawbore.schema.errors import SchemaCompatibilityError
from drawbore.tools import ToolAccessError
from drawbore.tools import registry as default_registry

from .catalog import AgentCatalog, resolve_ref
from .errors import ConfigResolutionError
from .fingerprint import schema_fingerprint
from .models import AgentConfig, NodeConfig, PipelineConfig, SCHEMA_VERSION, StepConfig


def from_json(
    data: "str | bytes | Mapping[str, Any]",
    *,
    agents: "AgentCatalog | Mapping[str, Agent]",
    registry: Any = None,
) -> Pipeline:
    """Parse a JSON document (str/bytes/mapping) into a ``PipelineConfig`` and
    resolve it. Malformed JSON and Pydantic validation failures (incl. unknown
    fields) both surface as ``ConfigResolutionError`` — ``from_json`` raises
    nothing else (fail closed)."""
    if isinstance(data, (str, bytes, bytearray)):
        try:
            raw = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ConfigResolutionError(f"config is not valid JSON: {exc}") from exc
    else:
        raw = dict(data)
    try:
        config = PipelineConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigResolutionError(f"invalid config document: {exc}") from exc
    return from_config(config, agents=agents, registry=registry)


def from_config(
    config: PipelineConfig,
    *,
    agents: "AgentCatalog | Mapping[str, Agent]",
    registry: Any = None,
) -> Pipeline:
    """Resolve a parsed manifest into a live ``Pipeline`` (fail-closed)."""
    if config.schema_version != SCHEMA_VERSION:
        raise ConfigResolutionError(f"unknown config schema_version {config.schema_version}")

    reg = registry if registry is not None else default_registry

    # Resolve + drift-check every agent declaration; index by manifest name.
    _reject_duplicate_agent_names(config)
    resolved: dict[str, Agent] = {}
    for ac in config.agents:
        agent = resolve_ref(agents, ac.ref)
        _check_declaration(ac, agent)
        # model + tools + fallback_model is accepted at import. Execution enforces
        # the safe loop fallback rules (pre-tool fallback may proceed; post-tool
        # failure fails closed). Drift, unknown tools, malformed fingerprints, and
        # literal bindings are still rejected below.
        _check_tools_registered(ac, reg)
        resolved[ac.name] = agent

    escalation = _escalation(config)
    pipeline = Pipeline(
        name=config.pipeline.name,
        version=config.pipeline.version,
        registry=reg,
        on_failure=escalation,
        confidence_threshold=config.pipeline.confidence_threshold,
    )

    for idx, node in enumerate(config.steps):
        if node.kind == "join":
            # all_present joins build their output model from `inputs` bindings and the
            # output model is INDEPENDENT of any source's model. Recovering from
            # sources[0] would silently use the WRONG model — fail CLOSED instead.
            if node.policy == "all_present":
                raise ConfigResolutionError(
                    f"join '{node.name}': all_present joins are not supported for "
                    f"JSON config round-trip (the join output model is built "
                    f"from bindings and cannot be recovered from the manifest)"
                )
            # exactly_one / first_by_priority: the manifest stores the join's output
            # schema as data only; rebuilding a live `Join` from JSON-schema alone is
            # not possible. Recover the `output` model from the first source's resolved
            # output model — selecting policies share a source output model and the
            # sources were drift-checked as agents above.
            if node.sources[0] not in resolved:
                raise ConfigResolutionError(
                    f"step {idx} (join '{node.name}') references unknown source "
                    f"'{node.sources[0]}'"
                )
            out_model = resolved[node.sources[0]].spec.output
            inputs = (
                {f: From(b.from_) for f, b in node.bindings.items()}
                if node.bindings
                else None
            )
            try:
                pipeline.add(
                    Join(
                        node.name,
                        sources=list(node.sources),
                        policy=node.policy,
                        output=out_model,
                        inputs=inputs,
                    )
                )
            except (SchemaCompatibilityError, ToolAccessError, ValueError, ValidationError) as exc:
                raise ConfigResolutionError(f"step {idx} (join '{node.name}'): {exc}") from exc
            continue

        step = node
        if step.agent not in resolved:
            raise ConfigResolutionError(
                f"step {idx} references unknown agent '{step.agent}'"
            )
        depends_on = _validate_input_mode(idx, step, config.steps)
        inputs = _build_inputs(step)
        when = (
            When(**step.when.model_dump())
            if step.when is not None
            else None
        )
        # Reconstruct the EvidencePolicy and add the step inside ONE try: from_config
        # must raise nothing but ConfigResolutionError. Even though EvidencePolicyConfig
        # mirrors EvidencePolicy today, a future validator divergence must not escape
        # raw — fail closed (ValidationError included).
        try:
            evidence = (
                EvidencePolicy(**step.evidence.model_dump())
                if step.evidence is not None
                else None
            )
            pipeline.add(
                resolved[step.agent], inputs=inputs,
                depends_on=depends_on, evidence=evidence, when=when,
            )
        except (SchemaCompatibilityError, ToolAccessError, ValueError, ValidationError) as exc:
            raise ConfigResolutionError(f"step {idx} ('{step.agent}'): {exc}") from exc

    return pipeline


def _reject_duplicate_agent_names(config: PipelineConfig) -> None:
    seen: set[str] = set()
    for ac in config.agents:
        if ac.name in seen:
            raise ConfigResolutionError(f"duplicate agent name '{ac.name}' in manifest")
        seen.add(ac.name)


def _check_declaration(ac: AgentConfig, agent: Agent) -> None:
    spec = agent.spec
    if spec.name != ac.name:
        raise ConfigResolutionError(
            f"agent ref '{ac.ref}' resolves to '{spec.name}', config declares '{ac.name}'"
        )
    if spec.version != ac.version:
        raise ConfigResolutionError(
            f"agent '{ac.name}' version drift: config {ac.version}, resolved {spec.version}"
        )
    if spec.risk_tier != ac.risk_tier:
        raise ConfigResolutionError(
            f"agent '{ac.name}' risk-tier drift: config {ac.risk_tier}, resolved {spec.risk_tier}"
        )
    if spec.requires_human_approval != ac.requires_human_approval:
        raise ConfigResolutionError(
            f"agent '{ac.name}' requires_human_approval drift: config "
            f"{ac.requires_human_approval}, resolved {spec.requires_human_approval}"
        )
    if spec.context_access != ac.context_access:
        raise ConfigResolutionError(
            f"agent '{ac.name}' context_access drift: config {ac.context_access}, "
            f"resolved {spec.context_access}"
        )
    if list(spec.tools) != list(ac.tools):
        raise ConfigResolutionError(
            f"agent '{ac.name}' tool drift: config {list(ac.tools)}, resolved {list(spec.tools)}"
        )
    if spec.model != ac.model:
        raise ConfigResolutionError(
            f"agent '{ac.name}' model drift: config {ac.model}, resolved {spec.model}"
        )
    if spec.fallback_model != ac.fallback_model:
        raise ConfigResolutionError(
            f"agent '{ac.name}' fallback_model drift: config {ac.fallback_model}, "
            f"resolved {spec.fallback_model}"
        )
    if spec.instructions != ac.instructions:
        raise ConfigResolutionError(f"agent '{ac.name}' instructions drift")
    if schema_fingerprint(spec.input) != ac.input_schema_hash:
        raise ConfigResolutionError(f"agent '{ac.name}' input schema drift")
    if schema_fingerprint(spec.output) != ac.output_schema_hash:
        raise ConfigResolutionError(f"agent '{ac.name}' output schema drift")


def _check_tools_registered(ac: AgentConfig, reg: Any) -> None:
    for tool in ac.tools:
        if not reg.has(tool):
            raise ConfigResolutionError(
                f"agent '{ac.name}' declared tool '{tool}' is not registered"
            )


def _escalation(config: PipelineConfig) -> EscalationPolicy | None:
    of = config.pipeline.on_failure
    if of is None:
        return None
    return EscalationPolicy(channel=of.channel, target=of.target, mode=of.mode)


def _validate_input_mode(idx: int, step: StepConfig, steps: list[StepConfig]) -> list[str]:
    """Validate the tagged input mode against the step's index + depends_on and
    return the canonical depends_on to use when adding the step."""
    inp = step.input
    if inp.bindings is not None:
        sources = sorted({b.from_.split(".", 1)[0] for b in inp.bindings.values()})
        if sorted(step.depends_on) != sources:
            raise ConfigResolutionError(
                f"step {idx} depends_on {step.depends_on} does not match its binding "
                f"sources {sources}"
            )
        return sources
    if inp.source == "initial":
        if idx != 0 or step.depends_on:
            raise ConfigResolutionError(
                f"step {idx} source='initial' is only valid for the first step with "
                f"depends_on=[]"
            )
        return []
    # source == "previous"
    if idx == 0:
        raise ConfigResolutionError("step 0 cannot use source='previous'")
    prev = steps[idx - 1]
    predecessor = _node_name(prev)
    if inp.agent != predecessor:
        raise ConfigResolutionError(
            f"step {idx} source='previous' must name the immediate predecessor "
            f"'{predecessor}', got '{inp.agent}'"
        )
    if list(step.depends_on) != [predecessor]:
        raise ConfigResolutionError(
            f"step {idx} source='previous' requires depends_on=['{predecessor}'], "
            f"got {step.depends_on}"
        )
    return [predecessor]


def _node_name(node: NodeConfig) -> str:
    return node.name if node.kind == "join" else node.agent


def _build_inputs(step: StepConfig) -> dict[str, From]:
    if step.input.bindings is None:
        return {}
    return {field: From(b.from_) for field, b in step.input.bindings.items()}
