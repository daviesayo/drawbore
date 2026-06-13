"""The tool proxy — the single chokepoint for every tool call.

No tool call reaches its handler except through ``ToolProxy.invoke``. The proxy
validates the single-use capability token, enforces tool-level scope, logs the
call (input/output hashes, duration), and trips a per-agent-step circuit breaker.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

from opentelemetry.trace import Status, StatusCode

from ..errors import CircuitBreakerError, TaintError, ToolAccessError, ToolError, TokenError
from ..registry import ToolRegistry
from ..taint import TaintLedger, TrustLabel
from ..tokens import CapabilityToken, TokenIssuer


def _denial_label(exc: "ToolError") -> str:
    if isinstance(exc, TokenError):
        return "denied:token"
    if isinstance(exc, CircuitBreakerError):
        return "denied:breaker"
    if isinstance(exc, ToolAccessError):
        return "denied:scope"
    if isinstance(exc, TaintError):
        return "denied:taint"
    return "denied"


class ToolProxy:
    def __init__(
        self,
        registry: ToolRegistry,
        issuer: TokenIssuer,
        max_calls_per_tool: int = 3,
        clock: Callable[[], float] = time.monotonic,
        *,
        ledger: "TaintLedger | None" = None,
        effect_ledger: "EffectLedger | None" = None,
    ):
        self._registry = registry
        self._issuer = issuer
        self._max = max_calls_per_tool
        self._clock = clock
        self._ledger = ledger if ledger is not None else TaintLedger()
        # Durable effect ledger (exactly-once replay on resume). None ⇒ no
        # cross-restart effect tracking, exactly as a missing CheckpointStore
        # means no cross-restart resume. Never a toggle for the guarantee.
        self._effect_ledger = effect_ledger
        # Per-run, per-step monotonic ordinal of effectful calls (the replay
        # match position on resume). Lives on the proxy beside the breaker
        # counts, since the proxy is constructed per run.
        self._effect_cursor: dict[tuple[str, int], int] = {}
        # Circuit breaker is per (run, agent-step, tool) — audit H1.
        self._counts: dict[tuple[str, Any, str], int] = {}
        self.log: list[dict[str, Any]] = []

    def effects_consumed(self, run_id: str, step: int) -> int:
        """Number of effectful calls the proxy has processed for ``(run_id,
        step)`` — the cursor value used by the pipeline's step-end orphan check."""
        return self._effect_cursor.get((run_id, step), 0)

    async def invoke(
        self,
        tool_ref: str,
        args: Any,
        token: CapabilityToken,
        run_ctx: Any,
        operation: str = "invoke",
    ) -> Any:
        # Deferred to call time to avoid an import cycle: ``drawbore.tools`` is
        # imported during ``drawbore.errors`` initialisation (errors -> tools.errors
        # -> tools.__init__ -> proxy), and ``drawbore.observability`` imports back
        # into ``drawbore.errors``. By call time all modules are fully initialised.
        from drawbore.observability import genai_span, payload_hash, semconv
        from drawbore.state.effect_ledger import (
            EffectDivergenceError,
            EffectEntry,
            EffectLedgerWriteError,
            EffectStatus,
            EffectUnresolvedError,
            ledger_args_hash,
        )

        from ..access import _reset_idempotency_key, _set_idempotency_key

        run_id = run_ctx.run_id
        step = getattr(run_ctx, "step", None)
        key = (run_id, step, tool_ref)
        start = self._clock()
        # Hash the input once and reuse it for both the span tag and the log entry
        # (one identity scheme; avoids divergence between span and log).
        input_hash = payload_hash(args)
        with genai_span(semconv.OP_EXECUTE_TOOL, tool_ref, {
            semconv.GEN_AI_TOOL_NAME: tool_ref,
            semconv.DRAWBORE_RUN_ID: run_id,
            semconv.DRAWBORE_STEP: step,
            semconv.DRAWBORE_TOOL_OPERATION: operation,
            semconv.DRAWBORE_INPUT_HASH: input_hash,
        }) as span:
            try:
                # 0. Resolve the tool first — the taint gate needs its facts.
                tool = self._registry.get(tool_ref)
                # 1. Sink-gate: refuse an exfil-capable call while the step's
                #    taint scope is UNTRUSTED, BEFORE consuming the token or counting
                #    the breaker — the denial is side-effect-free.
                if tool.exfil_capable and self._ledger.scope(run_id, step) is TrustLabel.UNTRUSTED:
                    raise TaintError(
                        f"exfil-capable tool '{tool_ref}' blocked under untrusted scope "
                        f"in step {step} of run '{run_id}'"
                    )
                # 2. Bump: invoking an UNTRUSTED-source tool taints the step (the
                #    static declared fact, not the return value).
                #    Deliberately before token/operation authz: a denied call to an
                #    untrusted-source tool still taints the step — over-taint on a
                #    denial is fail-safe (can only halt, never leak).
                if tool.source_trust is TrustLabel.UNTRUSTED:
                    self._ledger.observe(run_id, step, TrustLabel.UNTRUSTED)
                # 3. Validate + consume the single-use token (scope incl. operation,
                #    expiry). Invalid tokens never reach the breaker counter — audit H2.
                self._issuer.consume(token, tool_ref, run_id, operation)
                # 4. Operation policy.
                if operation not in tool.allowed_operations:
                    raise ToolAccessError(
                        f"operation '{operation}' is not allowed for tool '{tool_ref}'"
                    )
                # 5. Circuit breaker — count only authorised calls, per agent-step.
                self._counts[key] = self._counts.get(key, 0) + 1
                if self._counts[key] > self._max:
                    raise CircuitBreakerError(
                        f"tool '{tool_ref}' called more than {self._max} times by "
                        f"step {step} in run '{run_id}'"
                    )
                # 6. Execute. The effect ledger wraps ONLY the handler call;
                #    every gate above ran first and unchanged, so a denied call
                #    never leaves a phantom pending entry and the taint/token/
                #    breaker guarantees hold identically on resume.
                if self._effect_ledger is None or tool.effectful is False:
                    # Read-only / no-ledger path: unchanged, cursor not advanced.
                    result = await tool.handler(args)
                else:
                    effect_hash = ledger_args_hash(args)
                    pos = self._effect_cursor.get((run_id, step), 0)
                    self._effect_cursor[(run_id, step)] = pos + 1
                    entry = self._effect_ledger.entry_at(run_id, step, pos)
                    if entry is None:
                        # Fresh effectful call: record pending → fire → succeed.
                        ikey = hashlib.sha256(
                            json.dumps(
                                [run_id, str(step), str(pos), tool_ref, effect_hash]
                            ).encode()
                        ).hexdigest()
                        try:
                            self._effect_ledger.record_pending(
                                EffectEntry(
                                    run_id=run_id,
                                    step=step,
                                    position=pos,
                                    tool_ref=tool_ref,
                                    input_hash=effect_hash,
                                    idempotency_key=ikey,
                                    status=EffectStatus.PENDING,
                                    output=None,
                                )
                            )
                        except Exception as e:
                            # Pending write failed: halt BEFORE the handler fires
                            # (else a resume sees no entry and double-fires).
                            raise EffectLedgerWriteError(
                                f"failed to record pending effect for tool "
                                f"'{tool_ref}' at step {step} position {pos}"
                            ) from e
                        key_token = _set_idempotency_key(ikey)
                        try:
                            result = await tool.handler(args)
                            try:
                                self._effect_ledger.record_succeeded(
                                    run_id, step, pos, result
                                )
                            except Exception as e:
                                raise EffectLedgerWriteError(
                                    f"failed to record succeeded effect for tool "
                                    f"'{tool_ref}' at step {step} position {pos}"
                                ) from e
                        finally:
                            # Reset on every exit path so no stale key leaks.
                            _reset_idempotency_key(key_token)
                    elif entry.status == EffectStatus.SUCCEEDED and (
                        entry.tool_ref,
                        entry.input_hash,
                    ) == (tool_ref, effect_hash):
                        # Replay: return the recorded output, never re-fire the
                        # handler — but still emit a span/log so it's observable.
                        output_hash = payload_hash(entry.output)
                        span.set_attribute(semconv.DRAWBORE_OUTPUT_HASH, output_hash)
                        span.set_attribute(semconv.DRAWBORE_STATUS, "replay")
                        span.set_status(Status(StatusCode.OK))
                        self._log(
                            tool_ref, run_id, operation, input_hash,
                            output_hash, start, "replay",
                        )
                        return entry.output
                    elif entry.status == EffectStatus.SUCCEEDED:
                        raise EffectDivergenceError(
                            f"step {step} position {pos}: expected "
                            f"{entry.tool_ref}/{entry.input_hash}, got "
                            f"{tool_ref}/{effect_hash}"
                        )
                    else:
                        raise EffectUnresolvedError(
                            f"step {step} position {pos}: pending effect "
                            f"{entry.tool_ref}, idempotency_key={entry.idempotency_key}"
                        )
            except ToolError as exc:
                span.set_attribute(semconv.DRAWBORE_STATUS, _denial_label(exc))
                self._log(tool_ref, run_id, operation, input_hash, None, start, _denial_label(exc))
                raise
            except Exception:
                span.set_attribute(semconv.DRAWBORE_STATUS, "error")
                self._log(tool_ref, run_id, operation, input_hash, None, start, "error")
                raise
            output_hash = payload_hash(result)
            span.set_attribute(semconv.DRAWBORE_OUTPUT_HASH, output_hash)
            span.set_attribute(semconv.DRAWBORE_STATUS, "ok")
            span.set_status(Status(StatusCode.OK))
            self._log(tool_ref, run_id, operation, input_hash, output_hash, start, "ok")
            return result

    def _log(
        self,
        tool_ref: str,
        run_id: str,
        operation: str,
        input_hash: str,
        output_hash: str | None,
        start: float,
        label: str,
    ) -> None:
        # input_hash is computed once by invoke and passed in — no re-hash here.
        self.log.append(
            {
                "tool": tool_ref,
                "run_id": run_id,
                "operation": operation,
                "input_hash": input_hash,
                "output_hash": output_hash,
                "duration": self._clock() - start,
                "result": label,
            }
        )
