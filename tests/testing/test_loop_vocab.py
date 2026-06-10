import json
import pytest

from drawbore.testing import call, final, text, multi_call
from drawbore.testing.loop import to_turns


def test_call_converts_to_a_call_tuple():
    assert to_turns([call("lookup", {"id": "x"})]) == [("call", "lookup", {"id": "x"})]


def test_final_converts_to_a_json_text_tuple():
    turns = to_turns([final({"risk": "low"})])
    assert turns == [("text", json.dumps({"risk": "low"}, sort_keys=True))]
    # the loop parses this text as the final JSON answer
    assert json.loads(turns[0][1]) == {"risk": "low"}


def test_text_converts_to_a_text_tuple():
    assert to_turns([text("not json")]) == [("text", "not json")]


def test_multi_call_converts_to_a_multicall_tuple():
    turns = to_turns([multi_call(call("a", {}), call("b", {"k": 1}))])
    assert turns == [("multicall", [("a", {}), ("b", {"k": 1})])]


def test_a_mixed_script_converts_in_order():
    turns = to_turns([call("lookup", {"id": "1"}), final({"ok": True})])
    assert turns == [
        ("call", "lookup", {"id": "1"}),
        ("text", json.dumps({"ok": True}, sort_keys=True)),
    ]


def test_multi_call_requires_call_items():
    with pytest.raises(TypeError):
        multi_call(("a", {}))   # must be call(...) objects, not raw tuples
