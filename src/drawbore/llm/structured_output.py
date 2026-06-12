"""The one structured-output boundary every model turn passes through.

Every model turn — a one-shot final answer, a tool-loop final answer, and a
tool-loop tool-call turn — reduces to the same problem: turn the model's raw text
into a JSON object Drawbore can validate, or fail closed. This module is that one
mechanism, in exactly three stages:

  (a) deterministic extraction — :func:`extract_object`, pure, no I/O;
  (b) at most ONE bounded corrective reprompt — budget-gated, never two;
  (c) halt — :func:`structured_output_halt` raises ``LLMError`` (``model_error``)
      carrying a bounded excerpt of the offending text.

It is provider- and engine-agnostic: it imports only stdlib and
:mod:`drawbore.llm.errors`. The actual re-ask is supplied by the call site as an
``async`` adapter (a litellm re-call in the gateway; one more ADK session pass in
the loop), so the mechanism spans both subsystems without importing either.

Invariant: the mechanism NEVER parses or executes a tool out of prose. It only ever
yields a final JSON object; a tool runs only when the model emits a genuine
structured call, which is handled entirely outside this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Optional, Protocol

from .errors import LLMError, content_excerpt


@dataclass(frozen=True)
class Extraction:
    """The outcome of stage (a). ``kind`` is one of:

    - ``"object"`` — ``value`` is a usable JSON object (parsed directly, or the
      single top-level object deterministically recovered from prose);
    - ``"structural"`` — the text was parseable JSON that is NOT an object (an array
      or a scalar). A reprompt would not fix the model's intent: halt immediately;
    - ``"absent"`` — empty / null / non-JSON prose, or zero / more-than-one
      (ambiguous) top-level objects. Reprompt-eligible (one bounded re-ask).
    """

    kind: Literal["object", "structural", "absent"]
    value: dict | None
    detail: str


def _object(value: dict) -> Extraction:
    return Extraction(kind="object", value=value, detail="")


def _structural(detail: str) -> Extraction:
    return Extraction(kind="structural", value=None, detail=detail)


def _absent(detail: str) -> Extraction:
    return Extraction(kind="absent", value=None, detail=detail)


def extract_object(text: object) -> Extraction:
    """Stage (a): deterministically reduce raw model text to an :class:`Extraction`.

    Pure and side-effect-free (safe to run after tools have executed). Generalizes
    ``json.loads`` plus the deterministic sole-top-level-object recovery: it accepts
    a direct JSON object or a single prose-wrapped object, refuses non-object JSON
    (``structural``), and treats empty / null / non-JSON / ambiguous content as
    ``absent`` (reprompt-eligible).
    """
    if text is None:
        return _absent("the model returned null content (no text)")
    if not isinstance(text, str):
        return _absent(f"the model returned non-text content: {type(text).__name__}")
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        recovered = _sole_top_level_json_object(text)
        if recovered is not None:
            return _object(recovered)
        return _absent("the content was not a JSON object")
    if isinstance(parsed, dict):
        return _object(parsed)
    return _structural(f"the content was JSON but not an object: {type(parsed).__name__}")


def reprompt_instruction(*, tool_names: tuple[str, ...] = ()) -> str:
    """The corrective instruction for an ``absent`` outcome (prose / no usable JSON).

    With tools it restates that the model must emit EITHER a single structured tool
    call OR a final JSON object — it never parses or executes anything out of prose;
    the model must emit a real structured call for any tool to run. With no tools it
    asks only for a single JSON object.
    """
    if tool_names:
        names = ", ".join(sorted(tool_names))
        return (
            "Your previous message was prose and contained neither a structured tool "
            "call nor a final answer. Respond now with EITHER a single structured tool "
            "call OR the final answer as one JSON object, and nothing else — do not "
            "narrate, explain, or add any text outside the JSON. "
            f"The available tools are: {names}."
        )
    return (
        "Your previous message was not a single JSON object. Respond now with ONLY a "
        "single JSON object that matches the required schema — no prose, no "
        "explanation, and no text outside the JSON."
    )


def schema_reprompt_instruction(field_errors: str, *, tool_names: tuple[str, ...] = ()) -> str:
    """The corrective instruction for an object that PARSED but failed the agent's
    output schema. Seeded with the rendered field errors so the model knows exactly
    what to fix. The model still gets one chance only; the framework's schema gate is
    the authority on whether the corrected answer is accepted."""
    tool_clause = ""
    if tool_names:
        names = ", ".join(sorted(tool_names))
        tool_clause = f" The available tools are: {names}."
    return (
        "Your previous answer was a JSON object but it did not match the required "
        f"schema: {field_errors}. Respond now with ONLY a corrected single JSON object "
        "that fixes these problems and matches the schema — no prose and no text "
        f"outside the JSON.{tool_clause}"
    )


def structured_output_halt(agent: str, ex: Extraction, *, text: object, reprompted: bool) -> LLMError:
    """Stage (c): the ``model_error`` halt for an unrecoverable turn.

    Always carries a bounded excerpt of the offending text so the failure is
    diagnosable from the halt reason and audit record. Raising an ``LLMError`` keeps
    the gateway's fallback rule intact (a contract violation never advances the
    provider chain). The reason names the failure but no provider, engine, or
    internal type."""
    excerpt = content_excerpt(text)
    if ex.kind == "structural":
        return LLMError(
            f"agent {agent!r} returned JSON that is not a usable object: {ex.detail} "
            f"(content excerpt: {excerpt})"
        )
    suffix = " even after a single corrective reprompt" if reprompted else ""
    return LLMError(
        f"agent {agent!r} returned no usable JSON object{suffix}: {ex.detail} "
        f"(content excerpt: {excerpt})"
    )


class Budget(Protocol):
    """Whether the call site still has room for the single bounded re-ask. The driver
    additionally caps reprompts at one regardless of what this reports."""

    def can_reask(self) -> bool: ...


class OneShotBudget:
    """The one-shot gateway budget: a re-ask is always permitted because the driver's
    own one-reprompt cap enforces the historic single retry (one initial call + at
    most one re-ask)."""

    def can_reask(self) -> bool:
        return True


@dataclass(frozen=True)
class Coerced:
    """A validated JSON object plus the number of corrective reprompts it took
    (0 or 1). ``reprompts`` is surfaced to the audit trail."""

    value: dict
    reprompts: int


async def coerce_structured_output(
    *,
    agent: str,
    initial_text: object,
    reask: Callable[[str], Awaitable[object]],
    budget: Budget,
    tool_names: tuple[str, ...] = (),
    validate_object: Optional[Callable[[dict], Optional[str]]] = None,
) -> Coerced:
    """Run stages (a) -> (b) -> (c) for one model turn.

    Returns a :class:`Coerced` (a usable JSON object + the reprompt count) or raises
    ``LLMError`` (``model_error``) with a bounded excerpt. Never executes a tool.

    ``reask`` performs ONE more model turn for the given instruction and returns its
    raw text (the call site owns that I/O — a litellm re-call, or one more ADK session
    pass). ``budget`` gates the re-ask; the driver itself never reprompts more than
    once. ``tool_names`` seed the corrective instruction. ``validate_object``, when
    given (the loop path), reports the agent's output-schema errors for an extracted
    object as a rendered string (or ``None`` when valid); a schema-invalid object
    earns the SAME single bounded reprompt as an absent turn. The mechanism never
    raises ``schema_violation`` itself — when the reprompt budget is spent it returns
    the (still-invalid) object so the executor's authoritative output-schema gate is
    the one that halts.
    """
    reprompts = 0
    text = initial_text
    ex = extract_object(text)
    while True:
        if ex.kind == "object":
            errors = validate_object(ex.value) if validate_object is not None else None
            if errors is None:
                return Coerced(value=ex.value, reprompts=reprompts)
            # An object that parsed but failed the output schema. Reprompt-eligible,
            # sharing the single-reprompt budget. When the budget is spent, return it
            # so the executor's authoritative schema gate halts — never raise here.
            if reprompts >= 1 or not budget.can_reask():
                return Coerced(value=ex.value, reprompts=reprompts)
            text = await reask(schema_reprompt_instruction(errors, tool_names=tool_names))
            reprompts += 1
            ex = extract_object(text)
            continue
        if ex.kind == "structural":
            raise structured_output_halt(agent, ex, text=text, reprompted=reprompts > 0)
        # absent: reprompt-eligible
        if reprompts >= 1 or not budget.can_reask():
            raise structured_output_halt(agent, ex, text=text, reprompted=reprompts > 0)
        text = await reask(reprompt_instruction(tool_names=tool_names))
        reprompts += 1
        ex = extract_object(text)
        continue


def _sole_top_level_json_object(text: object) -> dict | None:
    """Deterministically recover a single top-level JSON OBJECT from prose-wrapped
    content.

    Returns the object iff EXACTLY ONE balanced top-level ``{...}`` span parses as a
    JSON object; returns ``None`` when the content is not a string, holds no such
    object, or holds more than one (ambiguous — there is no safe way to choose which
    to trust). Pure and bounded: a single left-to-right scan, no tool call, no model
    call, no side effect."""
    if not isinstance(text, str):
        return None
    found: list[dict] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            end = _matching_brace(text, i)
            if end != -1:
                try:
                    value = json.loads(text[i:end])
                except ValueError:
                    value = None
                if isinstance(value, dict):
                    found.append(value)
                    if len(found) > 1:
                        return None      # ambiguous: more than one top-level object
                i = end                  # skip the whole span (never descend into nested)
                continue
        i += 1
    return found[0] if len(found) == 1 else None


def _matching_brace(s: str, start: int) -> int:
    """Index just past the ``}`` that balances the ``{`` at ``start``, honouring JSON
    string literals and escapes so braces inside strings do not miscount; ``-1`` when
    the brace is unbalanced."""
    depth = 0
    in_str = False
    escaped = False
    for j in range(start, len(s)):
        c = s[j]
        if in_str:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return -1
