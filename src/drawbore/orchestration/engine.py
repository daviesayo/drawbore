"""Orchestration engine abstraction.

``OrchestratorEngine`` is the seam that keeps the orchestration engine invisible
to developers. It runs a single agent step; Drawbore's ``pipeline`` owns
cross-step policy. Concrete subclasses adapt a specific engine; ADK is imported
only inside this package's adapter, never elsewhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from drawbore.agent import AgentSpec

if TYPE_CHECKING:
    # Deferred: drawbore.llm transitively imports litellm. Annotation-only use is
    # safe under `from __future__ import annotations`; a future
    # get_type_hints(StepExecution) call in a cold-import context would NameError.
    from drawbore.llm import ModelAudit


@dataclass
class ToolLoopBundle:
    """What a model-capable engine needs to drive an in-step tool loop.

    Carries the proxy/issuer/registry the engine uses to build proxy-backed tools
    (the in-process ``ToolContext`` deliberately hides these so the engine cannot
    extract them from it), the agent's ``declared`` tool refs, and the ``run_ctx``
    for token issuance. Three list slots are MUTABLE ACCUMULATORS the loop mutates
    in place and clears between fallback attempts: ``failures`` (any in-loop tool
    failure — drives immediate abort), ``turns`` (one marker per model turn; the
    model-turn count is ``len(turns)``), and ``tool_invoked`` (one marker per tool
    the loop is about to run — lets the loop driver tell a pre-tool from a
    post-tool provider failure). This is deliberately a plain mutable dataclass:
    a ``frozen=True`` here would advertise immutability the accumulators violate.
    """

    proxy: Any
    issuer: Any
    registry: Any
    declared: tuple[str, ...]
    run_ctx: Any
    failures: list = field(default_factory=list)
    turns: list = field(default_factory=list)
    tool_invoked: list = field(default_factory=list)


@dataclass(frozen=True)
class StepExecution:
    """A model-capable engine's step result. ``output`` is the raw pre-validation
    value the pipeline validates; ``model_audit`` carries the provider-attempt
    summary (None for deterministic steps); ``model_turns`` is the model-turn count
    (0 deterministic, 1 one-shot, len(turns) for a loop). The pipeline normalizes a
    raw (non-StepExecution) return to ``StepExecution(output=value)`` for backward
    compatibility."""

    output: Any
    model_audit: ModelAudit | None = None
    model_turns: int = 0


class OrchestratorEngine(ABC):
    """Abstract base class for the per-step execution engine."""

    @abstractmethod
    async def run_step(
        self, spec: AgentSpec, payload: Any, tools: Any = None, *, tool_loop: ToolLoopBundle | None = None
    ) -> Any:
        """Execute one agent step and return its raw (pre-validation) output.

        The caller (the pipeline) validates the returned value against the agent's
        declared output schema. ``tools`` is a ``ToolContext`` passed only to
        in-process agents that declare tools. ``tool_loop`` is passed only for a
        model-backed agent that declares tools, so a model-capable engine can drive
        the proxy-scoped tool loop; engines with no model path ignore it.
        """
        ...
