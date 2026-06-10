"""A scripted fake loop model for local test mode.

A ``BaseLlm`` (an ADK type) that yields SCRIPTED turns instead of calling a live
model, so ``drawbore.testing`` can drive the real ADK ``Runner`` loop (proxy-backed
tools, callbacks, failure semantics, model-turn audit) with no network. It lives
here because ``google`` may be imported only under ``drawbore.orchestration``
(containment invariant); ``drawbore.testing`` receives only the opaque
``Callable[[str], BaseLlm]`` factory and never imports ADK.

Turn vocabulary (normalized tuples; ``drawbore.testing.loop`` builds these):
  ("call", name, args)             -> one function call
  ("multicall", [(name, args), …]) -> >1 function call in a turn (loop fails closed)
  ("text", text)                   -> a text turn (final answer when it is JSON)
Past the end of the script the model yields ``{}`` (an empty JSON object) as a
text turn. The loop treats a text turn as a final answer, so it terminates and
``{}`` is validated against the agent's output model — for an output model with
required fields this fails output validation and the step halts. (Genuine
``max_llm_calls`` exhaustion is reached by a script of only call turns that
never emits a final answer.)
"""

from __future__ import annotations

from typing import Any, Callable

from google.adk.models import BaseLlm, LlmResponse
from google.genai import types


class ScriptedLoopModel(BaseLlm):
    """Yields scripted ADK turns. Constructed per step by the test factory."""

    def __init__(self, turns: list[tuple]) -> None:
        super().__init__(model="drawbore-test-fake")
        # BaseLlm is a pydantic model — keep non-field state off the pydantic surface.
        object.__setattr__(self, "_turns", list(turns))
        object.__setattr__(self, "_i", 0)

    async def generate_content_async(self, llm_request, stream: bool = False):
        i = self._i
        object.__setattr__(self, "_i", i + 1)
        if i >= len(self._turns):
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="{}")])
            )
            return
        turn = self._turns[i]
        kind = turn[0]
        if kind == "call":
            _, name, args = turn
            yield LlmResponse(content=types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
            ))
        elif kind == "multicall":
            _, calls = turn
            parts = [
                types.Part(function_call=types.FunctionCall(name=n, args=a))
                for n, a in calls
            ]
            yield LlmResponse(content=types.Content(role="model", parts=parts))
        else:  # "text"
            _, text = turn
            yield LlmResponse(content=types.Content(
                role="model", parts=[types.Part(text=text)],
            ))


def make_scripted_model_factory(turns: list[tuple]) -> Callable[[str], Any]:
    """Return a ``model_factory(model_name) -> ScriptedLoopModel`` for
    ``ADKEngine(model_factory=...)``. The factory ignores the model name — the
    harness already scoped these ``turns`` to one agent by ``spec.name``."""

    def factory(model_name: str) -> Any:
        return ScriptedLoopModel(turns)

    return factory
