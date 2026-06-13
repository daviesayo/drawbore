"""The model-call gateway.

``LLMGateway`` is the ABC; ``LiteLLMGateway`` is the first implementation, calling
LiteLLM's async completion across a non-streaming fallback chain. Bifrost (the
regulated upgrade path) would be another implementation of the same ABC.

Non-streaming only: Drawbore-owned buffer-and-replay streaming continuity is
deferred. Fallback is request-time completion-with-fallback, NOT mid-stream replay.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import litellm

from .errors import LLMError, ModelUnavailableError
from .request import ModelRequest, ModelResponse
from .structured_output import OneShotBudget, coerce_structured_output
from .usage import extract_cost, extract_usage


def configure_provider_logging() -> None:
    """Silence the provider SDK's unsolicited debug printing.

    The provider SDK prints diagnostic blurbs (a "Provider List" pointer and a
    "Get Help" note) straight to stdout around failing model attempts, which
    drowns a program's own output. The gateway owns its dependency's logging
    config, so a framework user never has to reach past this boundary to quiet
    the provider SDK. Idempotent; touches only the provider's own debug printing,
    never any Drawbore safety, audit, or observability signal.
    """
    litellm.suppress_debug_info = True


class LLMGateway(ABC):
    """The model-call boundary. Implementations perform a single non-streaming
    completion for a :class:`ModelRequest`, walking its fallback chain."""

    @abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete ``request`` and return the parsed :class:`ModelResponse`."""


class LiteLLMGateway(LLMGateway):
    """Completes via ``litellm.acompletion`` (non-streaming), trying each model in
    the request's chain in order and advancing on failure."""

    def __init__(self) -> None:
        configure_provider_logging()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ]
        last_error: Exception | None = None
        for model in request.model_chain:
            try:
                return await self._complete_one(model, messages)
            except LLMError:
                # A contract violation (incl. a retryable empty/non-JSON body whose
                # single retry was already spent in _complete_one) is NOT a transport
                # failure — fail closed, never advance the chain.
                raise
            except Exception as exc:  # provider/network error → try the next model
                last_error = exc
                continue
        raise ModelUnavailableError(
            f"all models failed for chain {request.model_chain}: {last_error}"
        )

    async def _complete_one(self, model: str, messages: list[dict]) -> ModelResponse:
        """One model, routed through the shared structured-output boundary: a single
        bounded CORRECTIVE reprompt on an absent body (empty / null / non-JSON), an
        immediate fail-closed on structural JSON (an array/scalar), then halt. The
        reprompt re-calls the SAME model once with an added corrective instruction; it
        never advances the fallback chain (that is for transport failures) and never
        disables halt-and-escalate. A provider/transport exception propagates so the
        chain may fall back."""
        _won_response: object = None
        _won_content: object = None

        async def _reask(hint: str) -> object:
            nonlocal _won_response, _won_content
            corrective = messages + [{"role": "user", "content": hint}]
            response = await litellm.acompletion(model=model, messages=corrective, stream=False)
            content = self._content_or_halt(response, model)
            _won_response, _won_content = response, content
            return content

        response = await litellm.acompletion(model=model, messages=messages, stream=False)
        content = self._content_or_halt(response, model)
        _won_response, _won_content = response, content
        coerced = await coerce_structured_output(
            agent=model, initial_text=content, reask=_reask, budget=OneShotBudget(),
        )
        won, won_content = _won_response, _won_content
        return ModelResponse(
            output=coerced.value, model_used=model,
            raw_text=won_content if isinstance(won_content, str) else "",
            usage=extract_usage(won), cost=extract_cost(won),
            reprompts=coerced.reprompts,
        )

    @staticmethod
    def _content_or_halt(response, model: str) -> object:
        """Read ``choices[0].message.content`` or fail closed on an unreadable provider
        response shape (a retry would not fix it — never fall back)."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"model '{model}' returned an unexpected response shape: {exc}"
            ) from exc


# Quiet the provider SDK as soon as the model boundary is imported, so even a
# transitive import of this package (before any gateway is constructed) does not
# leave the provider's stdout debug printing on.
configure_provider_logging()
