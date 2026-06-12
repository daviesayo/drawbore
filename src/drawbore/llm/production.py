"""Single-call LiteLLM provider adapter. The runtime owns fallback, classification,
and the ModelAudit; this gateway does ONE provider call, applies per-provider config
(base_url/timeout/extra), and parses the 200. Contract failures (non-JSON / bad shape /
non-object) raise ``LLMError`` (model_error, no fallback); provider/transport exceptions
propagate RAW for the runtime to classify. ``LiteLLMGateway`` is unchanged and remains
the small direct-chain gateway.
"""

from __future__ import annotations

import json

import litellm

from .config import LLMRuntimeConfig
from .errors import LLMConfigError, LLMError
from .gateway import LLMGateway
from .request import ModelRequest, ModelResponse
from .usage import extract_cost, extract_usage


class ProductionLLMGateway(LLMGateway):
    """One non-streaming completion for ``request.model_chain[0]``."""

    def __init__(self, *, config: LLMRuntimeConfig) -> None:
        self._config = config

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if not request.model_chain:
            # A single-call adapter cannot invent a model; an empty chain is a
            # caller/config fault, not a provider failure — fail closed legibly so
            # the runtime does not misclassify a raw IndexError as a transport error.
            raise LLMConfigError("ProductionLLMGateway received an empty model_chain")
        model = request.model_chain[0]
        provider = model.split("/", 1)[0] if "/" in model else None
        kwargs: dict[str, object] = {
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
                kwargs["base_url"] = provider_cfg.base_url
            if provider_cfg.timeout_seconds is not None:
                kwargs["timeout"] = provider_cfg.timeout_seconds
            for k, v in provider_cfg.extra.items():
                kwargs.setdefault(k, v)
        # Provider/transport exceptions propagate raw (the runtime classifies them).
        response = await litellm.acompletion(**kwargs)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"model '{model}' returned an unexpected response shape: {exc}"
            ) from exc
        if content is None:
            # Distinct from a non-JSON string: the model returned no text at all
            # (e.g. a tool-use or malformed/partial response). Label it precisely.
            raise LLMError(
                f"model '{model}' returned null content "
                f"(no text in choices[0].message.content)"
            )
        try:
            output = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMError(f"model '{model}' returned non-JSON content: {exc}") from exc
        if not isinstance(output, dict):
            raise LLMError(
                f"model '{model}' returned JSON that is not an object: {type(output).__name__}"
            )
        return ModelResponse(
            output=output, model_used=model, raw_text=content,
            usage=extract_usage(response), cost=extract_cost(response),
        )
