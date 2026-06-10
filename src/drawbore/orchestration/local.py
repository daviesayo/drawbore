"""In-process engine for deterministic agents — no ADK, no model."""

from __future__ import annotations

from typing import Any

from drawbore.agent import AgentSpec

from .engine import OrchestratorEngine, ToolLoopBundle
from .errors import EngineError


class LocalEngine(OrchestratorEngine):
    """Runs a deterministic agent function directly in-process. It has NO model
    path: a model-backed agent (``spec.model is not None``) run here fails closed
    rather than silently executing ``spec.fn`` — the error is engine-agnostic and
    never names ADK."""

    async def run_step(
        self, spec: AgentSpec, payload: Any, tools: Any = None, *, tool_loop: ToolLoopBundle | None = None
    ) -> Any:
        if spec.model is not None:
            raise EngineError(
                f"agent '{spec.name}' is model-backed, but this engine has no model "
                f"path; use a model-capable OrchestratorEngine"
            )
        if spec.tools:
            return await spec.fn(payload, tools)
        return await spec.fn(payload)
