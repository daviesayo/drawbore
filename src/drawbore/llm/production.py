"""Single-call LiteLLM provider adapter. The runtime owns fallback, classification,
and the ModelAudit; this gateway does ONE provider call (plus, on a side-effect-free
contract violation, a single bounded retry of that same call), applies per-provider
config (base_url/timeout/extra), and parses the 200. Contract failures (non-JSON /
bad shape / non-object) raise ``LLMError`` (model_error, no fallback); provider/
transport exceptions propagate RAW for the runtime to classify. ``LiteLLMGateway`` is
the small direct-chain gateway with the same content-contract retry.
"""

from __future__ import annotations

import litellm

from .config import LLMRuntimeConfig
from .errors import LLMConfigError, LLMError
from .gateway import LLMGateway, configure_provider_logging
from .request import ModelRequest, ModelResponse
from .structured_output import OneShotBudget, coerce_structured_output
from .usage import extract_cost, extract_usage


class ProductionLLMGateway(LLMGateway):
    """One non-streaming completion for ``request.model_chain[0]``."""

    def __init__(self, *, config: LLMRuntimeConfig) -> None:
        configure_provider_logging()
        self._config = config

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if not request.model_chain:
            # A single-call adapter cannot invent a model; an empty chain is a
            # caller/config fault, not a provider failure — fail closed legibly so
            # the runtime does not misclassify a raw IndexError as a transport error.
            raise LLMConfigError("ProductionLLMGateway received an empty model_chain")
        model = request.model_chain[0]
        provider = model.split("/", 1)[0] if "/" in model else None
        base_kwargs: dict[str, object] = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        }
        provider_cfg = self._config.providers.get(provider) if provider else None
        if provider_cfg is not None:
            if provider_cfg.base_url is not None:
                base_kwargs["base_url"] = provider_cfg.base_url
            if provider_cfg.timeout_seconds is not None:
                base_kwargs["timeout"] = provider_cfg.timeout_seconds
            for k, v in provider_cfg.extra.items():
                base_kwargs.setdefault(k, v)
        # Provider-native structured output: when the runtime resolved a native-enabled
        # provider it set request.output_format to the agent's output class. Assign it
        # DIRECTLY into base_kwargs (NOT as a one-off kwarg at the call) and AFTER the
        # extra loop so it wins unconditionally AND the _reask closure's dict(base_kwargs)
        # inherits it — otherwise the corrective reprompt would silently lose native
        # constraint. Gate only on output_format (the runtime sets it under native only).
        if request.output_format is not None:
            base_kwargs["response_format"] = request.output_format
        # Route the 200 through the shared structured-output boundary: a single bounded
        # CORRECTIVE reprompt on a side-effect-free absent body (null / empty / non-JSON
        # — the transient class a re-ask routinely clears), an immediate fail-closed on
        # structural JSON (an array/scalar), then halt. The reprompt re-calls the same
        # model once (re-applying provider config) and never disables halt-and-escalate.
        # Provider/transport exceptions propagate raw on every call (the runtime
        # classifies them).
        _won_response: object = None
        _won_content: object = None

        async def _reask(hint: str) -> object:
            nonlocal _won_response, _won_content
            kwargs = dict(base_kwargs)
            kwargs["messages"] = list(base_kwargs["messages"]) + [
                {"role": "user", "content": hint}
            ]
            response = await litellm.acompletion(**kwargs)
            content = self._content_or_halt(response, model)
            _won_response, _won_content = response, content
            return content

        response = await litellm.acompletion(**base_kwargs)
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
        response shape (a retry would not fix it). ``None`` content is NOT a shape
        error — it is an absent body the boundary reprompts on."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"model '{model}' returned an unexpected response shape: {exc}"
            ) from exc
