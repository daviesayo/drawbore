"""The test orchestration engine for local test mode.

``TestEngine`` is an ``OrchestratorEngine`` that fakes the MODEL boundary while
reusing the real per-step engine. For each step it inspects ``spec.name`` (the only
reliable routing key — the gateway sees only a ``ModelRequest`` and the loop factory
sees only a model name) and delegates to a per-step ``ADKEngine``:

- deterministic agent (no model): ``ADKEngine`` runs the agent's REAL ``fn``.
- one-shot model agent: an agent-scoped ``FakeGateway`` supplies the response.
- model+tools agent: an agent-scoped scripted ``model_factory`` drives the REAL
  ADK loop (proxy-backed tools, callbacks, failure semantics, audit).

Missing mocks fail closed with ``TestingError``. No second loop implementation
exists — the real ``ADKEngine``/``run_agentic_loop_chain`` runs (the runtime routes
the model+tools path through the per-attempt chain driver)."""

from __future__ import annotations

from typing import Any, Mapping

from drawbore.agent import AgentSpec
from drawbore.llm import CredentialChecker, LLMGateway, LLMRuntime, LLMRuntimeConfig
from drawbore.orchestration import ADKEngine, OrchestratorEngine, ToolLoopBundle, make_scripted_model_factory

from .credentials import StaticCredentialChecker
from .errors import TestingError
from .gateway import FakeGateway
from .loop import to_turns

# Distinguishes "no mock for this agent" from "mock present with value None" so the
# latter still fails closed. Module-level identity, compared within run_step.
_MISSING = object()


class _RaisingGateway(LLMGateway):
    """A gateway that must never be called (deterministic / model+tools paths don't
    use the gateway). If it is, that is a wiring bug — surface it, don't hide it."""

    async def complete(self, request: Any) -> Any:
        raise TestingError("internal: gateway invoked on a non-one-shot-model step")


class TestEngine(OrchestratorEngine):
    """Routes by ``spec.name``, delegates to a per-step ``ADKEngine``."""

    def __init__(
        self,
        *,
        model_responses: Mapping[str, Any],
        loop_scripts: Mapping[str, Any],
        max_llm_calls: int = 8,
        llm_config: LLMRuntimeConfig | None = None,
        credential_checker: CredentialChecker | None = None,
    ) -> None:
        self._model_responses = dict(model_responses)
        self._loop_scripts = dict(loop_scripts)
        self._max_llm_calls = max_llm_calls
        # The per-step LLMRuntime resolves profile:* refs for real (config + fake
        # credential checker) while the FakeGateway/scripted factory supply the mocked
        # turn — so a profile RESOLVES and records a real ModelAudit, yet NO provider is
        # ever called.
        self._llm_config = llm_config if llm_config is not None else LLMRuntimeConfig()
        self._credential_checker = (
            credential_checker if credential_checker is not None
            else StaticCredentialChecker(available=True)
        )

    async def run_step(
        self, spec: AgentSpec, payload: Any, tools: Any = None,
        *, tool_loop: ToolLoopBundle | None = None,
    ) -> Any:
        if spec.model is not None and spec.tools:
            if spec.name not in self._loop_scripts:
                raise TestingError(
                    f"no mock loop script for model+tools agent '{spec.name}' "
                    f"(add it to mock_loop_scripts=)"
                )
            factory = make_scripted_model_factory(to_turns(self._loop_scripts[spec.name]))
            # The runtime resolves the chain (so a profile-backed loop agent resolves via
            # config + fake checker) and drives the REAL ADK loop with the scripted model
            # factory; the scripted factory ignores the model name.
            runtime = LLMRuntime(
                config=self._llm_config, gateway=_RaisingGateway(),
                credential_checker=self._credential_checker, model_factory=factory,
            )
            delegate = ADKEngine(llm_runtime=runtime, max_llm_calls=self._max_llm_calls)
        elif spec.model is not None:
            # One-shot model: an agent-scoped fake gateway. A missing entry yields a
            # gateway that fails closed when complete() is called. The module-level
            # _MISSING sentinel keeps "key present with value None" failing closed.
            response = self._model_responses.get(spec.name, _MISSING)
            gateway = FakeGateway(
                agent_name=spec.name,
                response=None if response is _MISSING else response,
            )
            # The runtime resolves the declared model (profile:* or direct) for real,
            # then calls the FakeGateway for the mocked response — no provider call.
            runtime = LLMRuntime(
                config=self._llm_config, gateway=gateway,
                credential_checker=self._credential_checker,
            )
            delegate = ADKEngine(llm_runtime=runtime, max_llm_calls=self._max_llm_calls)
        else:
            # Deterministic agent: ADKEngine runs its real fn; gateway unused.
            runtime = LLMRuntime.from_gateway(_RaisingGateway())
            delegate = ADKEngine(llm_runtime=runtime, max_llm_calls=self._max_llm_calls)
        return await delegate.run_step(spec, payload, tools, tool_loop=tool_loop)
