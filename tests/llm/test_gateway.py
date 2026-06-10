import pytest
from drawbore.llm import (
    LiteLLMGateway, LLMGateway, ModelRequest, ModelResponse, ModelUnavailableError,
)


def _request(chain):
    return ModelRequest(system="emit json", user='{"amount": 1}', model_chain=chain)


def _fake_response(content: str):
    # Mimic the litellm response shape: response["choices"][0]["message"]["content"].
    return {"choices": [{"message": {"content": content}}]}


def test_gateway_is_abstract():
    with pytest.raises(TypeError):
        LLMGateway()


async def test_completes_with_the_primary_model(monkeypatch):
    calls = []

    async def fake_acompletion(*, model, messages, stream):
        calls.append(model)
        assert stream is False
        return _fake_response('{"risk": "low"}')

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    gw = LiteLLMGateway()
    resp = await gw.complete(_request(("m1", "m2")))
    assert isinstance(resp, ModelResponse)
    assert resp.output == {"risk": "low"}
    assert resp.model_used == "m1"
    assert calls == ["m1"]  # fallback not used when primary succeeds


async def test_falls_back_to_the_next_model_on_failure(monkeypatch):
    calls = []

    async def fake_acompletion(*, model, messages, stream):
        calls.append(model)
        if model == "m1":
            raise RuntimeError("primary down")
        return _fake_response('{"risk": "high"}')

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    gw = LiteLLMGateway()
    resp = await gw.complete(_request(("m1", "m2")))
    assert resp.output == {"risk": "high"}
    assert resp.model_used == "m2"
    assert calls == ["m1", "m2"]


async def test_raises_model_unavailable_when_whole_chain_fails(monkeypatch):
    async def fake_acompletion(*, model, messages, stream):
        raise RuntimeError("down")

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    gw = LiteLLMGateway()
    with pytest.raises(ModelUnavailableError):
        await gw.complete(_request(("m1", "m2")))


async def test_invalid_json_output_raises_llm_error(monkeypatch):
    from drawbore.llm import LLMError

    async def fake_acompletion(*, model, messages, stream):
        return _fake_response("not json at all")

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    gw = LiteLLMGateway()
    with pytest.raises(LLMError):
        await gw.complete(_request(("m1",)))


async def test_unexpected_response_shape_raises_llm_error(monkeypatch):
    from drawbore.llm import LLMError

    async def fake_acompletion(*, model, messages, stream):
        return {"choices": []}  # 200 but no message/content — unexpected shape

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    gw = LiteLLMGateway()
    with pytest.raises(LLMError):
        await gw.complete(_request(("m1",)))


async def test_non_object_json_raises_llm_error(monkeypatch):
    from drawbore.llm import LLMError

    async def fake_acompletion(*, model, messages, stream):
        return _fake_response("123")  # valid JSON, but an int, not an object

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    gw = LiteLLMGateway()
    with pytest.raises(LLMError):
        await gw.complete(_request(("m1",)))


async def test_sends_system_and_user_as_messages(monkeypatch):
    seen = {}

    async def fake_acompletion(*, model, messages, stream):
        seen["messages"] = messages
        return _fake_response('{"risk": "low"}')

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    gw = LiteLLMGateway()
    await gw.complete(ModelRequest(system="SYS", user="USR", model_chain=("m1",)))
    roles = [(m["role"], m["content"]) for m in seen["messages"]]
    assert ("system", "SYS") in roles
    assert ("user", "USR") in roles
