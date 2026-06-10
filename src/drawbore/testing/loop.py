"""The local-test-mode loop-script vocabulary.

A small, explicit set of frozen dataclasses describing scripted model turns for a
model+tools agent, plus ``to_turns`` which normalizes them to the tuple vocabulary
the orchestration scripted model consumes. Pure Python — no ADK here; the ADK fake
model lives under ``drawbore.orchestration`` (containment).

  call("lookup", {"id": "x"})      -> the model asks to call a tool
  final({"risk": "low"})           -> a text turn whose text is the final JSON answer
  text("not json")                 -> a raw text turn (used to test non-JSON finals)
  multi_call(call(...), call(...)) -> a single turn requesting >1 call (fails closed)
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Call:
    name: str
    args: dict


@dataclass(frozen=True)
class Final:
    output: dict


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class MultiCall:
    calls: tuple["Call", ...]


def call(name: str, args: dict | None = None) -> Call:
    return Call(name=name, args=dict(args or {}))


def final(output: dict) -> Final:
    return Final(output=dict(output))


def text(value: str) -> Text:
    return Text(text=value)


def multi_call(*calls: Call) -> MultiCall:
    for c in calls:
        if not isinstance(c, Call):
            raise TypeError("multi_call(...) takes call(...) objects")
    return MultiCall(calls=tuple(calls))


def to_turns(script) -> list[tuple]:
    """Normalize a list of vocabulary objects into the orchestration tuple form."""
    turns: list[tuple] = []
    for item in script:
        if isinstance(item, Call):
            turns.append(("call", item.name, item.args))
        elif isinstance(item, Final):
            turns.append(("text", json.dumps(item.output, sort_keys=True)))
        elif isinstance(item, Text):
            turns.append(("text", item.text))
        elif isinstance(item, MultiCall):
            turns.append(("multicall", [(c.name, c.args) for c in item.calls]))
        else:
            raise TypeError(f"unknown loop-script item: {item!r}")
    return turns
