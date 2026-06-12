"""The ADK-backed execution engine.

Three paths: a deterministic agent (no ``model``) runs in-process (== LocalEngine);
a one-shot model agent (``model``, no tools) is resolved and completed through the
injected ``LLMRuntime``; a model agent that DECLARES TOOLS is run through the
hidden ADK Runner loop, which owns the multi-turn model call. ADK is imported only
in this package.
"""

from __future__ import annotations

from typing import Any, Callable

from drawbore.agent import AgentSpec
from drawbore.llm import LLMError, LLMGateway, LLMRuntime
from drawbore.observability import genai_span, semconv

from .adk_loop import run_agentic_loop, run_agentic_loop_chain  # noqa: F401 (compat export)
from .engine import OrchestratorEngine, StepExecution, ToolLoopBundle


def _default_model_factory(name: str):
    # ADK's LiteLLM-backed model. Imported lazily so a non-loop ADKEngine that never
    # builds a model does not require the import at construction.
    from google.adk.models.lite_llm import LiteLlm
    return LiteLlm(model=name)


class ADKEngine(OrchestratorEngine):
    """Runs a single agent step (deterministic / one-shot model / tool loop).

    Consumes the single ``LLMRuntime``. For backward compatibility ``gateway=`` is
    accepted and wrapped via ``LLMRuntime.from_gateway`` — exactly one of
    ``llm_runtime`` / ``gateway`` must be given."""

    def __init__(
        self,
        llm_runtime: LLMRuntime | None = None,
        *,
        gateway: LLMGateway | None = None,
        model_factory: Callable[[str], Any] = _default_model_factory,
        max_llm_calls: int = 8,
    ):
        if (llm_runtime is None) == (gateway is None):
            raise ValueError("ADKEngine requires exactly one of llm_runtime= or gateway=")
        if max_llm_calls < 1:
            raise ValueError(f"max_llm_calls must be >= 1, got {max_llm_calls}")
        self._runtime = llm_runtime if llm_runtime is not None else LLMRuntime.from_gateway(gateway)
        self._model_factory = model_factory
        self._max_llm_calls = max_llm_calls

    async def run_step(
        self, spec: AgentSpec, payload: Any, tools: Any = None, *, tool_loop: ToolLoopBundle | None = None
    ) -> Any:
        if spec.model is None:
            # Deterministic agent — identical to LocalEngine.
            if spec.tools:
                return await spec.fn(payload, tools)
            return await spec.fn(payload)

        # Model-backed AND declares tools -> the hidden ADK Runner loop. The loop
        # resolves through the runtime, so a model+tools agent may declare a
        # fallback_model: a transport failure BEFORE any tool runs may advance, a
        # post-tool provider failure fails closed, and a tool failure/denial always
        # aborts (run_agentic_loop_chain enforces this).
        if spec.tools:
            if tool_loop is None:
                raise LLMError(
                    f"agent '{spec.name}' is a model+tools agent but no tool loop was "
                    f"provided by the pipeline"
                )
            factory = self._runtime.model_factory or self._model_factory
            chain = self._runtime.resolve(spec)
            result = await run_agentic_loop_chain(
                spec, payload, tool_loop=tool_loop, run_id=tool_loop.run_ctx.run_id,
                chain=chain, model_factory=factory, max_llm_calls=self._max_llm_calls,
            )
            return StepExecution(
                output=result.output, model_audit=result.model_audit,
                model_turns=result.model_turns,
            )

        # One-shot model agent: the runtime resolves the chain and walks provider
        # attempts; the agent fn is not invoked. The returned StepExecution.output
        # dict is validated by the pipeline.
        chain = self._runtime.resolve(spec)
        with genai_span(semconv.OP_CHAT, chain.attempts[0].request_model, {
            semconv.GEN_AI_REQUEST_MODEL: chain.attempts[0].request_model,
            semconv.DRAWBORE_DECLARED_MODEL: chain.declared[0],
        }) as span:
            response = await self._runtime.complete(spec, payload, chain)
            span.set_attribute(semconv.GEN_AI_RESPONSE_MODEL, response.model_used)
        return StepExecution(
            output=response.output, model_audit=response.audit, model_turns=1,
            usage=response.usage, cost=response.cost,
        )
