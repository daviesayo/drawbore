"""The typed JSON-config manifest. Pure Pydantic data models.

Every model sets ``extra="forbid"``: unknown fields at ANY level — top-level,
pipeline, agent, step, input, binding, evidence — fail closed. These are data only;
resolution/serialization logic lives in ``resolver``/``serialization``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)


SCHEMA_VERSION = 1


class OnFailureConfig(BaseModel):
    """Serialized ``EscalationPolicy`` (channel/target/mode)."""

    model_config = ConfigDict(extra="forbid")

    channel: str
    target: str
    mode: Literal["sync", "async"] = "sync"


class PipelineMetaConfig(BaseModel):
    """The ``pipeline`` object: identity + pipeline-level policy."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    confidence_threshold: float | None = None
    on_failure: OnFailureConfig | None = None


class AgentConfig(BaseModel):
    """One agent declaration: a stable symbolic ``ref`` plus the declaration the
    config was reviewed against. ``from_json`` requires the resolved ``AgentSpec`` to
    match these (drift fails closed)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    ref: str
    version: str
    risk_tier: Literal["low", "medium", "high", "critical"]
    requires_human_approval: bool
    context_access: Literal["none"]
    tools: list[str]
    model: str | None
    fallback_model: str | None
    instructions: str | None
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    input_schema_hash: str
    output_schema_hash: str


class EvidencePolicyConfig(BaseModel):
    """Serialized per-step ``EvidencePolicy``. Mirrors ``EvidencePolicy``'s
    fields 1:1 but adds ``extra="forbid"`` (config must be stricter than the runtime
    ``EvidencePolicy(**data)``). Stores POLICY only; never stores/handles."""

    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    enabled: bool = False
    mode: Literal["simulate", "compress"] = "compress"
    allowed_transforms: tuple[str, ...] = ("json_rows", "logs")
    min_tokens: int = 800
    max_output_tokens: int | None = None
    require_original_store: bool = True
    allow_full_retrieval: bool = False
    allow_search_retrieval: bool = True
    ttl_seconds: int | None = None
    strict: bool = False


class BindingConfig(BaseModel):
    """A single ``{"from": "agent.field"}`` binding. Literal bindings are not
    supported and are rejected before any other validation."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: str = Field(alias="from")

    @model_validator(mode="before")
    @classmethod
    def _reject_literal(cls, data: Any) -> Any:
        if isinstance(data, dict) and "literal" in data:
            raise ValueError("literal input bindings are not supported; bind fields with From(...) references")
        return data


class StepInputConfig(BaseModel):
    """A tagged input mode. Exactly one of:
    ``source="initial"`` / ``source="previous"`` (+``agent``) / ``bindings``. This
    model validates SHAPE only; index/predecessor/depends_on SEMANTICS are checked in
    the resolver and serializer (they need step position + the live topology)."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["initial", "previous"] | None = None
    agent: str | None = None
    bindings: dict[str, BindingConfig] | None = None

    @model_validator(mode="after")
    def _exactly_one_well_formed_mode(self) -> "StepInputConfig":
        if self.bindings is not None:
            if self.source is not None or self.agent is not None:
                raise ValueError("input cannot mix 'bindings' with 'source'/'agent'")
            return self
        if self.source == "initial":
            if self.agent is not None:
                raise ValueError("input source='initial' takes no 'agent'")
            return self
        if self.source == "previous":
            if self.agent is None:
                raise ValueError("input source='previous' requires 'agent'")
            return self
        raise ValueError(
            "input must be source='initial', source='previous' with 'agent', or 'bindings'"
        )


class ConditionConfig(BaseModel):
    """Serialized ``When`` branch condition. Mirrors ``When``'s leaf operators 1:1:
    exactly one of ``equals``/``in_``/``is_true``/``gt``/``lt`` against an
    ``agent.field`` ``ref``. Shape only; the live ``When`` re-validates the
    exactly-one-operator rule on resolve."""

    model_config = ConfigDict(extra="forbid")

    ref: str
    equals: Any | None = None
    in_: tuple[Any, ...] | None = None
    is_true: bool | None = None
    gt: float | None = None
    lt: float | None = None


class JoinConfig(BaseModel):
    """A serialized explicit ``Join`` node. Tagged ``kind="join"`` so the node union
    discriminates it from an agent ``StepConfig``. ``bindings`` is only populated for
    the ``all_present`` policy (From bindings); other policies forward a branch and
    carry no bindings."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["join"] = "join"
    name: str
    sources: list[str]
    policy: Literal["exactly_one", "first_by_priority", "all_present"]
    output_schema: dict[str, Any]
    output_schema_hash: str
    bindings: dict[str, BindingConfig] | None = None


class StepConfig(BaseModel):
    """One ordered agent step: which agent, its tagged input mode, its explicit
    ``depends_on`` (legibility + round-trip stability), an optional branch ``when``
    condition, and optional evidence policy.

    ``kind`` discriminates this from a ``JoinConfig`` in the node union.
    Its ``"agent"`` default is OMITTED on serialize so an agent-only pipeline's
    manifest is byte-identical to a kind-less manifest."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["agent"] = "agent"
    agent: str
    input: StepInputConfig
    depends_on: list[str] = Field(default_factory=list)
    when: ConditionConfig | None = None
    evidence: EvidencePolicyConfig | None = None

    @model_serializer(mode="wrap")
    def _ser(self, handler: Any) -> Any:
        # Drop the `kind="agent"` default on serialize so an agent-only pipeline's
        # manifest matches a kind-less manifest byte-for-byte (round-trip stability).
        data = handler(self)
        if data.get("kind") == "agent":
            data.pop("kind", None)
        return data


# A pipeline node is either an agent step or an explicit join, discriminated by
# ``kind``. The before-validator on ``PipelineConfig`` injects the ``"agent"``
# default for legacy kind-less nodes ahead of this dispatch.
NodeConfig = Annotated[
    Union[StepConfig, JoinConfig], Field(discriminator="kind")
]


class PipelineConfig(BaseModel):
    """The whole manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    pipeline: PipelineMetaConfig
    agents: list[AgentConfig]
    steps: list[NodeConfig]

    @model_validator(mode="before")
    @classmethod
    def _default_node_kind(cls, data: Any) -> Any:
        # Legacy manifests may have no `kind` on steps. Inject "agent" BEFORE the
        # discriminated-union dispatch so they load unchanged — Pydantic's
        # `Field(discriminator="kind")` otherwise treats `kind` as required and would
        # fail closed on a kind-less node.
        if isinstance(data, dict) and isinstance(data.get("steps"), list):
            new_steps = [
                {**node, "kind": "agent"} if isinstance(node, dict) and "kind" not in node else node
                for node in data["steps"]
            ]
            data = {**data, "steps": new_steps}
        return data
