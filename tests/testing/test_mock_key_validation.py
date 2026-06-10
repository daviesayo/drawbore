"""Eager construction-time validation of model/loop mock keys.

``pipeline.test_mode(...)`` already rejects a tool mock that names an undeclared tool
at construction (via ``build_scoped_registry``). These tests pin the symmetric guard
for ``mock_model_responses`` and ``mock_loop_scripts``: a key that does not name a
step of the matching category raises ``TestingError`` at ``test_mode(...)`` time —
before any ``run`` — instead of silently doing nothing until a run-time fail-closed
halt. ``mock_model_responses`` is consumed only for one-shot model agents (model set,
no tools); ``mock_loop_scripts`` only for model+tools agents.
"""

import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.testing import TestingError, final
from drawbore.tools import ToolRegistry


class In(BaseModel):
    x: int


class Out(BaseModel):
    x: int


def _one_shot_pipeline():
    @agent(name="score", input=In, output=Out, model="fake")
    async def score(v: In) -> Out: ...
    p = Pipeline(name="p", registry=ToolRegistry())
    p.add(score)
    return p


def _loop_pipeline():
    reg = ToolRegistry()

    async def lookup(args):
        return {"hit": True}

    reg.register_tool("lookup", lookup, allowed_operations=("invoke",), schema={"type": "object"})

    @agent(name="solve", input=In, output=Out, model="fake", tools=["lookup"])
    async def solve(v: In, tools) -> Out: ...
    p = Pipeline(name="p", registry=reg)
    p.add(solve)
    return p


def _deterministic_pipeline():
    @agent(name="fetch", input=In, output=Out)
    async def fetch(v: In) -> Out:
        return Out(x=v.x)
    p = Pipeline(name="p", registry=ToolRegistry())
    p.add(fetch)
    return p


def test_typo_model_response_key_raises_at_construction():
    p = _one_shot_pipeline()
    with pytest.raises(TestingError, match="scorer"):
        p.test_mode(mock_model_responses={"scorer": {"x": 1}})


def test_typo_loop_script_key_raises_at_construction():
    p = _loop_pipeline()
    with pytest.raises(TestingError, match="solv"):
        p.test_mode(mock_loop_scripts={"solv": [final({"x": 1})]})


def test_model_response_key_for_a_deterministic_agent_raises():
    p = _deterministic_pipeline()
    with pytest.raises(TestingError, match="fetch"):
        p.test_mode(mock_model_responses={"fetch": {"x": 1}})


def test_model_response_key_for_a_model_tools_agent_raises():
    # A model+tools agent's responses belong in mock_loop_scripts, not
    # mock_model_responses (consumed only for one-shot model agents) — a key here is
    # a dead entry, so it fails closed at construction.
    p = _loop_pipeline()
    with pytest.raises(TestingError, match="solve"):
        p.test_mode(mock_model_responses={"solve": {"x": 1}})


def test_loop_script_key_for_a_one_shot_agent_raises():
    p = _one_shot_pipeline()
    with pytest.raises(TestingError, match="score"):
        p.test_mode(mock_loop_scripts={"score": [final({"x": 1})]})


def test_valid_keys_construct_without_error():
    _one_shot_pipeline().test_mode(mock_model_responses={"score": {"x": 1}})
    _loop_pipeline().test_mode(mock_loop_scripts={"solve": [final({"x": 1})]})


def test_empty_mocks_construct_without_error():
    _one_shot_pipeline().test_mode()
    _loop_pipeline().test_mode()
