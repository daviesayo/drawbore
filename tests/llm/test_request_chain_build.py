import json
import pytest
from pydantic import BaseModel
from drawbore.agent import agent
from drawbore.llm import (
    ModelRequest, ModelResponse, resolve_model_chain, build_model_request, LLMError,
)


class In(BaseModel):
    amount: int


class Out(BaseModel):
    risk: str


def _spec(model=None, fallback=None, instructions=None):
    @agent(name="scorer", input=In, output=Out, model=model,
           fallback_model=fallback, instructions=instructions)
    async def fn(v: In) -> Out:
        return Out(risk="low")
    return fn.spec


def test_resolve_chain_is_model_then_fallback():
    assert resolve_model_chain(_spec(model="m1", fallback="m2")) == ("m1", "m2")


def test_resolve_chain_dedupes_and_drops_none():
    assert resolve_model_chain(_spec(model="m1", fallback="m1")) == ("m1",)
    assert resolve_model_chain(_spec(model="m1")) == ("m1",)


def test_resolve_chain_raises_when_no_model():
    with pytest.raises(LLMError):
        resolve_model_chain(_spec())


def test_build_model_request_is_explicit_and_carries_payload_and_schema():
    req = build_model_request(
        _spec(model="m1", instructions="Score it."),
        In(amount=100),
        model_chain=("m1",),
    )
    assert isinstance(req, ModelRequest)
    assert req.model_chain == ("m1",)
    assert "Score it." in req.system           # instructions preserved
    assert "risk" in req.system                 # output schema embedded as the JSON contract
    assert json.loads(req.user) == {"amount": 100}   # payload rendered as data, parseable
    assert "ONLY" in req.system                  # the JSON-only contract is stated verbatim
    assert "no code fences" in req.system         # parsing relies on no markdown fences
    assert not req.system.startswith("\n")        # instructions present → no stray leading blank line


def test_build_model_request_without_instructions_still_states_the_json_contract():
    req = build_model_request(_spec(model="m1"), In(amount=5), model_chain=("m1",))
    assert "JSON" in req.system or "json" in req.system


def test_model_response_holds_parsed_output():
    resp = ModelResponse(output={"risk": "high"}, model_used="m1", raw_text='{"risk": "high"}')
    assert resp.output == {"risk": "high"}
    assert resp.model_used == "m1"
