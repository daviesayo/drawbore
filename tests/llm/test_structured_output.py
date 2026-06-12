"""Unit tests for the pure structured-output boundary mechanism.

The mechanism reduces one model turn's raw text to a usable JSON object, allows at
most ONE bounded corrective reprompt, then halts model_error with a bounded excerpt.
It is provider- and engine-agnostic: no ADK, no litellm.
"""

import json

import pytest

from drawbore.llm import LLMError
from drawbore.llm.structured_output import (
    Coerced,
    OneShotBudget,
    coerce_structured_output,
    extract_object,
)


# --- stage (a): deterministic extraction ---

def test_extract_direct_object():
    ex = extract_object(json.dumps({"a": 1}))
    assert ex.kind == "object"
    assert ex.value == {"a": 1}


def test_extract_sole_prose_wrapped_object_is_recovered():
    text = 'Sure! Here is the result:\n{"answer": "ok"}\nLet me know.'
    ex = extract_object(text)
    assert ex.kind == "object"
    assert ex.value == {"answer": "ok"}


def test_extract_zero_objects_is_absent():
    assert extract_object("just prose, no json here").kind == "absent"


def test_extract_ambiguous_multiple_objects_is_absent():
    ex = extract_object('first {"a": 1} then {"b": 2}')
    assert ex.kind == "absent"          # ambiguous: never guess which to trust


def test_extract_non_object_json_is_structural():
    assert extract_object("123").kind == "structural"
    assert extract_object("[1, 2, 3]").kind == "structural"


def test_extract_empty_null_nonstring_is_absent():
    assert extract_object("").kind == "absent"
    assert extract_object(None).kind == "absent"
    assert extract_object(None).detail.lower().find("null") != -1   # legible "null content"
    assert extract_object(123).kind == "absent"


# --- stage (b)/(c): the async driver ---

async def _never_reask(_hint):
    raise AssertionError("reask must not be called")


async def test_object_on_first_try_no_reask():
    coerced = await coerce_structured_output(
        agent="m", initial_text=json.dumps({"x": 1}),
        reask=_never_reask, budget=OneShotBudget(),
    )
    assert coerced == Coerced(value={"x": 1}, reprompts=0)


async def test_absent_then_reask_then_object():
    async def reask(_hint):
        return json.dumps({"x": 2})

    coerced = await coerce_structured_output(
        agent="m", initial_text="prose, no json", reask=reask, budget=OneShotBudget(),
    )
    assert coerced.value == {"x": 2}
    assert coerced.reprompts == 1


async def test_absent_then_reask_still_absent_halts_with_excerpt():
    async def reask(_hint):
        return "still just prose"

    with pytest.raises(LLMError) as ei:
        await coerce_structured_output(
            agent="m", initial_text="prose one", reask=reask, budget=OneShotBudget(),
        )
    assert "content excerpt" in str(ei.value)


async def test_structural_halts_immediately_reask_not_called():
    with pytest.raises(LLMError):
        await coerce_structured_output(
            agent="m", initial_text="[1, 2, 3]",        # JSON, not an object
            reask=_never_reask, budget=OneShotBudget(),
        )


async def test_budget_exhausted_halts_without_reask():
    class _NoBudget:
        def can_reask(self):
            return False

    with pytest.raises(LLMError):
        await coerce_structured_output(
            agent="m", initial_text="prose", reask=_never_reask, budget=_NoBudget(),
        )


async def test_reprompt_is_bounded_to_one_even_when_budget_allows():
    # A budget that would always allow more must NOT produce a second reask: the
    # driver caps reprompts at one regardless of the budget.
    calls = []

    class _Generous:
        def can_reask(self):
            return True

    async def reask(_hint):
        calls.append(1)
        return "still not json"

    with pytest.raises(LLMError):
        await coerce_structured_output(
            agent="m", initial_text="prose", reask=reask, budget=_Generous(),
        )
    assert len(calls) == 1                       # exactly one reask, never two


# --- #29: output-schema reprompt (loop-side, via validate_object) ---

async def test_object_failing_schema_reprompts_then_recovers():
    seen_hint = {}

    async def reask(hint):
        seen_hint["hint"] = hint
        return json.dumps({"answer": "fixed"})

    def validate_object(obj):
        return None if "answer" in obj else "answer: Field required"

    coerced = await coerce_structured_output(
        agent="m", initial_text=json.dumps({"wrong": "x"}),
        reask=reask, budget=OneShotBudget(), validate_object=validate_object,
    )
    assert coerced.value == {"answer": "fixed"}
    assert coerced.reprompts == 1
    assert "answer: Field required" in seen_hint["hint"]   # field errors seed the reprompt


async def test_object_failing_schema_then_still_invalid_returns_object_not_raise():
    # The mechanism never raises schema_violation itself: after the single corrective
    # reprompt it RETURNS the (still-invalid) object so the executor's authoritative
    # output-schema gate is the one that halts schema_violation.
    async def reask(_hint):
        return json.dumps({"still": "bad"})

    def validate_object(obj):
        return None if "answer" in obj else "answer: Field required"

    coerced = await coerce_structured_output(
        agent="m", initial_text=json.dumps({"wrong": "x"}),
        reask=reask, budget=OneShotBudget(), validate_object=validate_object,
    )
    assert coerced.value == {"still": "bad"}
    assert coerced.reprompts == 1               # exactly one corrective chance, then yield


async def test_shared_budget_absent_reprompt_then_schema_invalid_object_does_not_stack():
    # An ABSENT-failure reprompt and a schema-failure reprompt SHARE one budget. An
    # absent first turn consumes the single reprompt; a schema-invalid answer after it
    # must NOT earn a second reprompt — the object is returned for the executor gate.
    calls = []

    async def reask(_hint):
        calls.append(1)
        return json.dumps({"wrong": "y"})        # an object, but schema-invalid

    def validate_object(obj):
        return None if "answer" in obj else "answer: Field required"

    coerced = await coerce_structured_output(
        agent="m", initial_text="prose, no json",
        reask=reask, budget=OneShotBudget(), validate_object=validate_object,
    )
    assert coerced.value == {"wrong": "y"}
    assert coerced.reprompts == 1
    assert len(calls) == 1                       # never a second reask
