"""The fake one-shot model gateway for local test mode.

A pure ``LLMGateway`` (NO ADK, NO provider SDK) built per step for ONE agent. The
harness already chose this gateway by ``spec.name``; the gateway must NOT infer the
agent from prompt text or model name. ``complete`` resolves the scripted value
(dict / Pydantic model / sequence consumed in order / callable(ModelRequest)) into a
production-shaped ``ModelResponse``. Missing or exhausted -> ``TestingError`` (fail
closed); an exception value is raised so the pipeline halts through the normal model
path."""

from __future__ import annotations

import inspect
import json
from typing import Any

from pydantic import BaseModel

from drawbore.llm import LLMGateway, ModelRequest, ModelResponse, TokenUsage

from .errors import TestingError


def _is_exception(value: Any) -> bool:
    return isinstance(value, BaseException) or (
        isinstance(value, type) and issubclass(value, BaseException)
    )


class FakeGateway(LLMGateway):
    """Returns scripted one-shot responses for a single named agent."""

    def __init__(
        self, *, agent_name: str, response: Any, usage: TokenUsage | None = None
    ) -> None:
        self._agent_name = agent_name
        # A list/tuple is a sequence consumed in order; anything else is reused.
        self._sequence = list(response) if isinstance(response, (list, tuple)) else None
        self._single = None if self._sequence is not None else response
        self._has_single = self._sequence is None and response is not None
        # The fake provider's reported token usage for this agent (None = the
        # provider reports no usage, mirroring a real call that omits it).
        self._usage = usage

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self._sequence is not None:
            if not self._sequence:
                raise TestingError(
                    f"no mock model response left for agent '{self._agent_name}' "
                    f"(sequence exhausted)"
                )
            value = self._sequence.pop(0)
        elif self._has_single:
            value = self._single
        else:
            raise TestingError(
                f"no mock model response for agent '{self._agent_name}'"
            )

        if _is_exception(value):
            raise value() if isinstance(value, type) else value
        if callable(value) and not isinstance(value, BaseModel):
            value = value(request)
            if inspect.isawaitable(value):
                value = await value
        output = value.model_dump() if isinstance(value, BaseModel) else value
        if not isinstance(output, dict):
            raise TestingError(
                f"mock model response for agent '{self._agent_name}' must resolve to a "
                f"dict (the output model JSON), got {type(output).__name__}"
            )
        return ModelResponse(
            output=output, model_used=request.model_chain[0],
            raw_text=json.dumps(output, sort_keys=True),
            usage=self._usage,
        )
