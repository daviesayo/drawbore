import json

import pytest

from drawbore.llm import (
    LLMConfigError,
    LLMError,
    LLMRuntimeConfig,
    ModelRequest,
    ProviderConfig,
)
from drawbore.llm.production import ProductionLLMGateway


class _FakeAcompletion:
    """Captures kwargs and returns a scripted litellm-shaped response or raises."""

    def __init__(self, *, result=None, raises=None):
        self.result = result
        self.raises = raises
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return self.result


class _ScriptedAcompletion:
    """Returns a per-call scripted response (or raises a per-call exception). After
    the script is exhausted it keeps returning the last item, so a 1-element script
    means 'the same response on every call'."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.results[min(len(self.calls) - 1, len(self.results) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def _resp(content):
    return {"choices": [{"message": {"content": content}}]}


def _req(model):
    return ModelRequest(system="sys", user="usr", model_chain=(model,))


async def test_single_call_parses_json_and_applies_provider_config(monkeypatch):
    fake = _FakeAcompletion(result=_resp(json.dumps({"y": 1})))
    monkeypatch.setattr("drawbore.llm.production.litellm.acompletion", fake)
    cfg = LLMRuntimeConfig(
        providers={"openrouter": ProviderConfig(base_url="https://x", timeout_seconds=12.0)}
    )
    gw = ProductionLLMGateway(config=cfg)
    resp = await gw.complete(_req("openrouter/anthropic/claude-3-5-sonnet"))
    assert resp.output == {"y": 1}
    assert resp.model_used == "openrouter/anthropic/claude-3-5-sonnet"
    assert fake.calls[0]["base_url"] == "https://x"
    assert fake.calls[0]["timeout"] == 12.0


async def test_non_json_content_is_a_contract_error(monkeypatch):
    fake = _FakeAcompletion(result=_resp("not json"))
    monkeypatch.setattr("drawbore.llm.production.litellm.acompletion", fake)
    gw = ProductionLLMGateway(config=LLMRuntimeConfig())
    with pytest.raises(LLMError):
        await gw.complete(_req("gpt-4o"))


async def test_provider_exception_propagates_unclassified(monkeypatch):
    boom = RuntimeError("provider down")
    fake = _FakeAcompletion(raises=boom)
    monkeypatch.setattr("drawbore.llm.production.litellm.acompletion", fake)
    gw = ProductionLLMGateway(config=LLMRuntimeConfig())
    with pytest.raises(RuntimeError):  # the runtime (Task 8) classifies; the gateway re-raises raw
        await gw.complete(_req("gpt-4o"))


async def test_null_content_is_a_contract_error(monkeypatch):
    # litellm can return None content (tool-use / partial response): fail closed
    # with a distinct, legible message, not a JSON-parse error.
    fake = _FakeAcompletion(result=_resp(None))
    monkeypatch.setattr("drawbore.llm.production.litellm.acompletion", fake)
    gw = ProductionLLMGateway(config=LLMRuntimeConfig())
    with pytest.raises(LLMError, match="null content"):
        await gw.complete(_req("gpt-4o"))


async def test_non_json_error_includes_bounded_content_excerpt(monkeypatch):
    junk = "<html>oops " + "X" * 9000 + "</html>"
    fake = _ScriptedAcompletion([_resp(junk)])
    monkeypatch.setattr("drawbore.llm.production.litellm.acompletion", fake)
    gw = ProductionLLMGateway(config=LLMRuntimeConfig())
    with pytest.raises(LLMError) as ei:
        await gw.complete(_req("gpt-4o"))
    msg = str(ei.value)
    assert "content excerpt" in msg
    assert "<html>oops" in msg
    assert len(msg) < 1000                   # bounded: the 9000-char body is not dumped


async def test_retries_once_on_contract_violation_then_succeeds(monkeypatch):
    fake = _ScriptedAcompletion([_resp(""), _resp(json.dumps({"y": 1}))])
    monkeypatch.setattr("drawbore.llm.production.litellm.acompletion", fake)
    gw = ProductionLLMGateway(config=LLMRuntimeConfig())
    resp = await gw.complete(_req("gpt-4o"))
    assert resp.output == {"y": 1}
    assert len(fake.calls) == 2              # one transient empty, then one retry that succeeded


async def test_second_contract_violation_fails_closed(monkeypatch):
    fake = _ScriptedAcompletion([_resp("nope"), _resp("still nope")])
    monkeypatch.setattr("drawbore.llm.production.litellm.acompletion", fake)
    gw = ProductionLLMGateway(config=LLMRuntimeConfig())
    with pytest.raises(LLMError) as ei:
        await gw.complete(_req("gpt-4o"))
    assert len(fake.calls) == 2              # exactly one retry, then halt — fail-closed preserved
    assert "content excerpt" in str(ei.value)


async def test_empty_model_chain_fails_closed_as_config_error():
    gw = ProductionLLMGateway(config=LLMRuntimeConfig())
    with pytest.raises(LLMConfigError):
        await gw.complete(ModelRequest(system="s", user="u", model_chain=()))
