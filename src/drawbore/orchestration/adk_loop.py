"""The ADK Runner-driven in-step tool loop. Imported only inside
``drawbore.orchestration``. The Runner/LlmAgent own the multi-turn model call for
a model-backed agent that declares tools; Drawbore owns the tools (proxy-backed),
the final-output parse + the failure semantics. ``build_model_request`` supplies
the prompt; the gateway is NOT used on this path.

The outer driver ``run_agentic_loop_chain`` wraps the single-pass
``_run_one_attempt``: a transport failure BEFORE any tool has run may advance to
the next attempt; once any tool has run a provider failure FAILS CLOSED (no replay
of tool side-effects across providers); a tool failure/denial aborts immediately
with the tool error (never a provider fallback)."""

from __future__ import annotations

import logging as _logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable

# Pin google_adk and opentelemetry.context to CRITICAL so ADK-internal execution-
# failure / cancellation log noise never leaks through the engine boundary (the
# ADK-hiding invariant). Drawbore surfaces all meaningful errors through its own
# typed exceptions; ADK's own logging is redundant. The OTel context logger is
# suppressed for the same reason: ADK's generator-cancel pattern (the hard-break
# causes runner.run_async to be abandoned mid-iteration) throws GeneratorExit into
# ADK's span context manager in a different asyncio context, triggering a cosmetic
# "Token was created in a different Context" detach error that does not affect
# span correctness or Drawbore's own span lifecycle.
_logging.getLogger("google_adk").setLevel(_logging.CRITICAL)
_logging.getLogger("opentelemetry.context").setLevel(_logging.CRITICAL)

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from drawbore.agent import AgentSpec
from drawbore.llm import (
    LLMError,
    ModelAttemptAudit,
    ModelAudit,
    ModelUnavailableError,
    build_model_request,
    resolve_model_chain,
)
from drawbore.llm.classify import classify_provider_exception
from drawbore.llm.resolution import ResolvedModelChain
from drawbore.llm.structured_output import coerce_structured_output
from drawbore.schema import render_schema_errors, validate
from drawbore.schema.errors import SchemaValidationError
from drawbore.tools.errors import ToolAccessError

from .adk_tools import (
    make_loop_before_tool_callback,
    make_loop_model_callbacks,
    proxy_backed_tool,
)
from .engine import ToolLoopBundle, provider_safe_tool_aliases

_APP = "drawbore"


@dataclass
class LoopBudget:
    """Whether the loop still has model-turn budget for the single bounded re-ask.
    ``remaining = max_llm_calls - len(turns)`` read live (``turns`` is the loop's
    mutable per-attempt turn accumulator), so the budget reflects turns consumed so
    far. The structured-output driver additionally caps reprompts at one."""

    max_llm_calls: int
    turns: list

    def can_reask(self) -> bool:
        return (self.max_llm_calls - len(self.turns)) >= 1


@dataclass(frozen=True)
class LoopResult:
    """The result of an agentic loop run over a (possibly multi-attempt) chain.
    ``output`` is the model's final JSON (the pipeline validates it); ``model_audit``
    carries the provider-attempt summary; ``model_turns`` is the model-turn count of
    the winning attempt; ``reprompts`` is how many bounded corrective structured-output
    reprompts the winning attempt took (0 or 1)."""

    output: dict
    model_audit: ModelAudit
    model_turns: int
    reprompts: int = 0


async def _run_one_attempt(
    spec: AgentSpec,
    payload: Any,
    *,
    tool_loop: ToolLoopBundle,
    run_id: str,
    model: str,
    model_factory: Callable[[str], Any],
    max_llm_calls: int,
) -> tuple[dict, int, int]:
    """Run a loop attempt on one concrete ``model`` and return
    ``(output_dict, model_turns, reprompts)``. ``output_dict`` is the model's final
    JSON (the pipeline validates it). Raises the captured tool failure (immediate
    abort) or ``LLMError`` on an unusable final answer / loop exhaustion. The chain
    driver calls this once per provider attempt.

    The first pass produces final text; the shared structured-output boundary then
    extracts a usable JSON object, allowing at most ONE bounded corrective reprompt on
    the SAME session before failing closed. Two recovery cases share that single
    budget: (1) the turn returned prose with no usable tool call or JSON answer (a
    narrated tool call is indistinguishable from a malformed answer — both surface as
    non-JSON text; Drawbore never parses or executes a tool out of prose, it re-asks
    for a structured response), and (2) the answer parsed as an object but failed the
    agent's output schema (the re-ask is seeded with the field errors). After the one
    re-ask, an object that still fails the schema is RETURNED so the pipeline's
    authoritative output-schema gate halts ``schema_violation``; an answer that is
    still not a usable object halts ``model_error``."""
    request = build_model_request(spec, payload, model_chain=(model,))

    # Expose each declared tool to the model under a provider-safe ALIAS (a function
    # name a provider can represent); the proxy-backed callable still invokes the
    # CANONICAL ref, so scope/JIT/audit are unchanged. The before-tool guard derives
    # the same aliases from the same declared tuple, so the two always agree.
    aliases = provider_safe_tool_aliases(tool_loop.declared)
    tools = [
        proxy_backed_tool(
            ref, name=aliases[ref], proxy=tool_loop.proxy, issuer=tool_loop.issuer,
            run_ctx=tool_loop.run_ctx, schema=_schema_of(tool_loop.registry, ref),
            failures=tool_loop.failures,
        )
        for ref in tool_loop.declared
    ]
    before_model, after_model = make_loop_model_callbacks(tool_loop, run_id=run_id)
    loop_agent = LlmAgent(
        name=spec.name,
        model=model_factory(model),
        instruction=request.system,
        tools=tools,
        before_model_callback=before_model,
        after_model_callback=after_model,
        before_tool_callback=make_loop_before_tool_callback(tool_loop),
    )

    session_service = InMemorySessionService()
    session_id = f"{run_id}:{tool_loop.run_ctx.step}:{uuid.uuid4().hex[:8]}"
    await session_service.create_session(app_name=_APP, user_id=run_id, session_id=session_id)
    runner = Runner(agent=loop_agent, app_name=_APP, session_service=session_service)

    # First pass: the model's original prompt.
    message = types.Content(role="user", parts=[types.Part(text=request.user)])
    final_text = await _consume_pass(
        spec, runner, run_id=run_id, session_id=session_id, message=message,
        tool_loop=tool_loop, max_llm_calls=max_llm_calls,
    )

    async def _reask(hint: str) -> object:
        # ONE more pass on the SAME session/runner for continuity, bounded by the
        # remaining model-turn budget. A pure model turn: no tool runs unless the model
        # emits a real structured call.
        remaining = max_llm_calls - len(tool_loop.turns)
        reprompt = types.Content(role="user", parts=[types.Part(text=hint)])
        return await _consume_pass(
            spec, runner, run_id=run_id, session_id=session_id, message=reprompt,
            tool_loop=tool_loop, max_llm_calls=remaining,
        )

    def _validate_object(obj: dict) -> str | None:
        # The reprompt-decision check: report the agent's output-schema errors so a
        # schema-invalid answer earns one corrective re-ask. This never replaces the
        # pipeline's authoritative post-step output-schema gate — a still-invalid
        # answer is returned for that gate to halt ``schema_violation``.
        try:
            validate(spec.output, obj)
            return None
        except SchemaValidationError as exc:
            return render_schema_errors(exc.errors)

    coerced = await coerce_structured_output(
        agent=spec.name, initial_text=final_text, reask=_reask,
        budget=LoopBudget(max_llm_calls, tool_loop.turns),
        tool_names=tuple(aliases.values()), validate_object=_validate_object,
    )
    return coerced.value, len(tool_loop.turns), coerced.reprompts


async def _consume_pass(
    spec: AgentSpec,
    runner: "Runner",
    *,
    run_id: str,
    session_id: str,
    message: "types.Content",
    tool_loop: ToolLoopBundle,
    max_llm_calls: int,
) -> str | None:
    """Drive ONE ``runner.run_async`` pass over ``message`` and return the model's final
    text (``None`` when the pass produced no final response — loop-call exhaustion).

    Raises the captured tool failure (immediate abort) or a multicall ``LLMError``; a raw
    provider/ADK exception is re-raised unwrapped so the chain driver can classify it.
    The same session/runner is reused across passes, so a reprompt continues the
    conversation with prior context; ``max_llm_calls`` bounds THIS pass."""
    final_text: str | None = None
    multicall_error: LLMError | None = None      # set if a turn emits >1 function call
    try:
        async for event in runner.run_async(
            user_id=run_id, session_id=session_id, new_message=message,
            run_config=RunConfig(max_llm_calls=max_llm_calls),
        ):
            if tool_loop.failures:
                break                            # HARD-BREAK: stop pulling events (primary guard)
            # Fail closed on a turn that REQUESTS more than one tool call:
            # detect it on the model-output event (before ADK dispatches the calls)
            # and stop pulling events so none of the requested tools execute. A
            # sentinel + break (not a raise here) is used because ADK consumes its
            # own iteration internally; the post-loop raise is the robust path.
            parts = event.content.parts if (event.content and event.content.parts) else []
            fcalls = [p for p in parts if getattr(p, "function_call", None)]
            if len(fcalls) > 1:
                multicall_error = LLMError(
                    f"agent '{spec.name}' loop emitted {len(fcalls)} function calls in one "
                    f"turn; Drawbore allows one call per turn (fail-closed)"
                )
                break
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text
    except Exception:
        # If a tool failure was captured, prefer the precise tool error.
        if tool_loop.failures:
            raise tool_loop.failures[0]
        # Otherwise re-raise whatever ADK/the model raised so the DRIVER can classify it
        # (a transport failure before any tool may fall back; ADK exhaustion → LLMError).
        # We do NOT wrap a provider/transport exception here, so the driver sees the
        # original and ``classify_provider_exception`` can read its concrete type.
        raise

    if tool_loop.failures:                       # captured but ADK did not raise — re-raise (backstop)
        raise tool_loop.failures[0]
    if multicall_error is not None:              # >1 tool call in one turn — fail closed
        raise multicall_error
    return final_text


async def run_agentic_loop(
    spec: AgentSpec,
    payload: Any,
    *,
    tool_loop: ToolLoopBundle,
    run_id: str,
    model_factory: Callable[[str], Any],
    max_llm_calls: int,
) -> tuple[dict, int]:
    """Single-attempt entrypoint (kept for compatibility). Resolves the direct chain
    (no fallback) and runs ONE pass on the primary model. Raises the captured tool
    failure (immediate abort) or ``LLMError`` on non-JSON / loop exhaustion. New
    code should use ``run_agentic_loop_chain``; the engine does.

    Returns ``(output_dict, model_turns)`` (the reprompt count is carried by the chain
    driver's ``LoopResult``, not this compat tuple). This compat wrapper ensures a raw
    ADK error (e.g. loop exhaustion) surfaces as an ``LLMError``. ``_run_one_attempt``
    re-raises the raw provider/ADK exception so the chain driver can classify it; here,
    where there is no chain to fall back on, any non-tool, non-``LLMError`` exception is
    wrapped."""
    chain = resolve_model_chain(spec)            # primary first; fallback rejected upstream
    try:
        output, turns, _reprompts = await _run_one_attempt(
            spec, payload, tool_loop=tool_loop, run_id=run_id, model=chain[0],
            model_factory=model_factory, max_llm_calls=max_llm_calls,
        )
        return output, turns
    except LLMError:
        raise                                    # already legible (non-JSON / exhaustion / multicall)
    except Exception as exc:
        # Belt-and-suspenders: ``_run_one_attempt`` already re-raises the captured tool
        # failure on this path, so this guard is harmless and purely defensive.
        if tool_loop.failures:                   # a tool failure/denial — surface it precisely
            raise tool_loop.failures[0]
        raise LLMError(f"agentic loop failed for agent '{spec.name}': {exc}") from exc


async def run_agentic_loop_chain(
    spec: AgentSpec,
    payload: Any,
    *,
    tool_loop: ToolLoopBundle,
    run_id: str,
    chain: ResolvedModelChain,
    model_factory: Callable[[str], Any],
    max_llm_calls: int,
) -> LoopResult:
    """Drive the loop across the resolved ``chain``. A pre-tool transport failure
    (whose reason is in the attempt's ``fallback_on``) may advance to the next
    attempt; once ANY tool has run, a provider failure FAILS CLOSED (no replay of
    tool side-effects across providers). A tool failure/denial aborts immediately
    (the inner pass re-raises the captured tool error — never a provider fallback)."""
    attempts: list[ModelAttemptAudit] = []
    for i, attempt in enumerate(chain.attempts):
        # Reset per-attempt model-turn / failure / tool markers for a clean pass.
        tool_loop.turns.clear()
        tool_loop.failures.clear()
        tool_loop.tool_invoked.clear()
        try:
            output, turns, reprompts = await _run_one_attempt(
                spec, payload, tool_loop=tool_loop, run_id=run_id,
                model=attempt.request_model, model_factory=model_factory,
                max_llm_calls=max_llm_calls,
            )
        except Exception as exc:
            # (Invariant 2) A captured TOOL failure/denial aborts immediately — never
            # a provider fallback. A denied tool must not let the model "retry" on
            # another provider to route around a tripped control. Checked FIRST,
            # before any provider classification.
            if tool_loop.failures:
                attempts.append(_loop_attempt_audit(i, attempt, "halted", reason="tool_failure"))
                raise

            # A Drawbore loop fail-closed signal (LLMError and subclasses: non-JSON,
            # multicall, no-final answer) is NOT a provider failure — re-raise as-is; it
            # already carries a legible ``model_error`` halt reason and never falls back.
            if isinstance(exc, LLMError):
                attempts.append(_loop_attempt_audit(i, attempt, "halted", reason="loop_error"))
                raise

            tool_ran = bool(tool_loop.tool_invoked)
            outcome = classify_provider_exception(exc)
            is_provider_failure = outcome.kind in ("transport", "auth", "contract")

            if not is_provider_failure:
                # An UNKNOWN/non-provider error (e.g. ADK's loop-call-limit exhaustion):
                # this is the loop's own budget firing, not a provider transport failure.
                # Wrap as an LLMError (``model_error``) and halt — never a fallback. Safe
                # post-tool: wrapping halts, it does not replay across providers.
                attempts.append(_loop_attempt_audit(i, attempt, "halted", reason="loop_error"))
                raise LLMError(
                    f"agentic loop failed for agent '{spec.name}': {exc}"
                ) from exc

            if tool_ran:
                # (Invariant 1) A genuine PROVIDER failure AFTER a tool ran: fail closed,
                # no replay of tool side-effects across providers. Never a fallback.
                attempts.append(_loop_attempt_audit(i, attempt, "halted", reason="after_tools"))
                raise ModelUnavailableError(
                    f"agent '{spec.name}' loop provider failed after a tool call; "
                    f"failing closed (post-tool fallback is unsafe without replay)"
                ) from exc

            if (
                outcome.kind == "transport"
                and outcome.reason in attempt.fallback_on
                and i + 1 < len(chain.attempts)
            ):
                # Pre-tool transport failure with a configured fallback reason and a
                # next attempt available: advance (the only safe fallback).
                attempts.append(_loop_attempt_audit(i, attempt, "fallback", reason=outcome.reason))
                continue

            # A pre-tool provider failure that is not advanceable (auth/contract, an
            # unconfigured transport reason, or the last attempt): fail closed.
            attempts.append(_loop_attempt_audit(
                i, attempt, "halted", reason=(outcome.reason or "provider_error")
            ))
            raise ModelUnavailableError(
                f"agent '{spec.name}' loop provider attempt {attempt.request_model} "
                f"failed and is not an advanceable fallback: {exc}"
            ) from exc
        else:
            attempts.append(_loop_attempt_audit(i, attempt, "success"))
            return LoopResult(
                output=output,
                model_audit=ModelAudit(
                    declared_refs=chain.declared,
                    selected_provider=attempt.provider,
                    selected_model=attempt.model,
                    attempts=tuple(attempts),
                    # DELIBERATE: on the success path, ``"not_loop"`` (i == 0) covers BOTH
                    # a one-shot agent and a loop agent that succeeded on the first attempt
                    # with no fallback; ``"before_tools"`` marks a success reached only after
                    # a pre-tool fallback advanced the chain. The reserved Literal
                    # ``"after_tools_halted"`` is intentionally NOT emitted here because a
                    # post-tool halt RAISES instead of returning (the pipeline then records
                    # model=None on the failed step) — test 17 asserts that case via the
                    # raised ModelUnavailableError + factory-call-count, not via this phase.
                    loop_fallback_phase=("before_tools" if i > 0 else "not_loop"),
                ),
                model_turns=turns,
                reprompts=reprompts,
            )
    # Defensive: only reached if ``chain.attempts`` is empty, which ``ResolvedModelChain``
    # guarantees against (it always carries at least one attempt). Each real attempt either
    # returns on success or raises on failure inside the loop.
    raise ModelUnavailableError(
        f"agent '{spec.name}' loop exhausted all model attempts"
    )


def _loop_attempt_audit(index, attempt, outcome, *, reason=None) -> ModelAttemptAudit:
    return ModelAttemptAudit(
        index=index, declared_ref=attempt.declared_ref, source=attempt.source,
        provider=attempt.provider, model=attempt.model,
        request_model=attempt.request_model, outcome=outcome, reason=reason,
    )


def _schema_of(registry, ref: str):
    # A declared-but-unregistered tool degrades to an open-object declaration
    # (``_schema_to_genai(None)``); any other error (a broken registry/``.schema``)
    # is a real bug and must propagate, not be silently swallowed.
    try:
        return registry.get(ref).schema
    except ToolAccessError:
        return None
