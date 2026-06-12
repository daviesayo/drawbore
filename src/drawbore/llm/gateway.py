"""The model-call gateway.

``LLMGateway`` is the ABC; ``LiteLLMGateway`` is the first implementation, calling
LiteLLM's async completion across a non-streaming fallback chain. Bifrost (the
regulated upgrade path) would be another implementation of the same ABC.

Non-streaming only: Drawbore-owned buffer-and-replay streaming continuity is
deferred. Fallback is request-time completion-with-fallback, NOT mid-stream replay.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

import litellm

from .errors import (
    LLMError,
    ModelUnavailableError,
    _RetryableContractError,
    content_excerpt,
)
from .request import ModelRequest, ModelResponse
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
        """One model, with a SINGLE bounded retry on a side-effect-free contract
        violation (empty / non-JSON body). The retry re-calls the SAME model once;
        it never advances the fallback chain (that is for transport failures) and it
        never disables halt-and-escalate — after the retry a still-bad body raises.
        A provider/transport exception propagates so the chain may fall back."""
        for attempt in range(2):  # one initial call + one contract retry
            response = await litellm.acompletion(model=model, messages=messages, stream=False)
            try:
                return self._parse(response, model)
            except _RetryableContractError:
                if attempt == 0:
                    continue  # transient empty/non-JSON 200: re-call the same model once
                raise         # still bad after the single retry: fail closed (model_error)

    def _parse(self, response, model: str) -> ModelResponse:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            # A 200 with an unexpected shape is a STRUCTURAL contract violation; a
            # retry would not fix it — fail closed immediately, never fall back.
            raise LLMError(
                f"model '{model}' returned an unexpected response shape: {exc}"
            ) from exc
        try:
            output = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            # A 200 with empty / non-JSON content: the transient class. Surface a
            # bounded excerpt so it is diagnosable, and mark it retryable.
            raise _RetryableContractError(
                f"model '{model}' returned non-JSON content: {exc} "
                f"(content excerpt: {content_excerpt(content)})"
            ) from exc
        if not isinstance(output, dict):
            # Parseable JSON of the wrong shape is structural, not transient — fail
            # closed immediately (no retry).
            raise LLMError(
                f"model '{model}' returned JSON that is not an object: {type(output).__name__}"
            )
        return ModelResponse(
            output=output, model_used=model, raw_text=content,
            usage=extract_usage(response), cost=extract_cost(response),
        )


# Quiet the provider SDK as soon as the model boundary is imported, so even a
# transitive import of this package (before any gateway is constructed) does not
# leave the provider's stdout debug printing on.
configure_provider_logging()
