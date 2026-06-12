"""The model+tools agentic loop must construct its provider model through the SAME
provider runtime config as the one-shot path: ``base_url``, ``timeout_seconds``, and
the ``ProviderConfig.extra`` pass-through (e.g. ``response_format`` json_object) must
reach the loop's model wrapper. Before the fix the default loop factory built the
model from the model name alone and dropped this config, so loop steps could not
honour provider-side JSON enforcement or transport settings that the one-shot
``ProductionLLMGateway`` already applies.

This drives the loop with a recording fake standing in for the ADK LiteLLM wrapper,
so the assertion inspects the construction kwargs and no provider is ever called.
"""

import json

from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.llm import LLMRuntime, LLMRuntimeConfig, ProviderConfig
from drawbore.orchestration import ADKEngine, ToolLoopBundle
from drawbore.tools import RunContext, TokenIssuer, ToolProxy, ToolRegistry


class In(BaseModel):
    task: str


class Out(BaseModel):
    answer: str


def _make_recording_litellm(records):
    """A ``BaseLlm`` subclass standing in for the ADK LiteLLM wrapper. Records its
    construction kwargs and yields a final JSON answer so the loop terminates without
    calling a provider. ADK is a core dep, so importing ``BaseLlm`` in a test is
    allowed (conftest does the same)."""
    from google.adk.models import BaseLlm, LlmResponse
    from google.genai import types

    class RecordingLiteLlm(BaseLlm):
        def __init__(self, **kwargs):
            super().__init__(model=kwargs.get("model", "rec"))
            records.append(dict(kwargs))

        async def generate_content_async(self, llm_request, stream=False):
            yield LlmResponse(content=types.Content(
                role="model", parts=[types.Part(text=json.dumps({"answer": "ok"}))]))

    return RecordingLiteLlm


def _registry():
    reg = ToolRegistry()

    async def _l(args):
        return {"v": 1}

    reg.register_tool(
        "lookup", _l, allowed_operations=("invoke",),
        schema={"type": "object", "properties": {}},
    )
    return reg


def _bundle(reg):
    issuer = TokenIssuer()
    return ToolLoopBundle(
        proxy=ToolProxy(reg, issuer), issuer=issuer, registry=reg,
        declared=("lookup",), run_ctx=RunContext(run_id="r1", step=0),
    )


def _loop_spec():
    @agent(
        name="solver", input=In, output=Out, tools=["lookup"],
        model="openrouter/anthropic/claude-3-5-sonnet",
    )
    async def solver(v: In) -> Out:
        raise AssertionError("must not run")

    return solver.spec


async def test_loop_default_factory_applies_provider_config(monkeypatch):
    # A direct provider-prefixed model string resolves with credential_required=False
    # (no credential preflight). The ProviderConfig keyed by the provider must reach
    # the loop's model wrapper, exactly as the one-shot gateway applies it.
    records: list[dict] = []
    monkeypatch.setattr(
        "google.adk.models.lite_llm.LiteLlm", _make_recording_litellm(records)
    )
    runtime = LLMRuntime(config=LLMRuntimeConfig(providers={
        "openrouter": ProviderConfig(
            base_url="https://gateway.example/v1",
            timeout_seconds=12.0,
            extra={"response_format": {"type": "json_object"}},
        )
    }))
    engine = ADKEngine(llm_runtime=runtime)  # DEFAULT factory — no override.
    reg = _registry()

    execution = await engine.run_step(
        _loop_spec(), In(task="t"), tool_loop=_bundle(reg)
    )

    assert execution.output == {"answer": "ok"}
    assert len(records) == 1
    built = records[0]
    assert built["model"] == "openrouter/anthropic/claude-3-5-sonnet"
    assert built["base_url"] == "https://gateway.example/v1"
    assert built["timeout"] == 12.0
    assert built["response_format"] == {"type": "json_object"}
