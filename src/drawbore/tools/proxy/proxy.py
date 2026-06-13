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


#: Default per-step circuit-breaker cap (also the ``ToolProxy.__init__`` default).
DEFAULT_MAX_CALLS_PER_TOOL: int = 3


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
        max_tool_calls_per_run: int = 500,
        max_distinct_tools_per_run: int = 50,
    ):
        if max_tool_calls_per_run < 1:
            raise ValueError(
                f"max_tool_calls_per_run must be >= 1, got {max_tool_calls_per_run}"
            )
        if max_distinct_tools_per_run < 1:
            raise ValueError(
                f"max_distinct_tools_per_run must be >= 1, got {max_distinct_tools_per_run}"
            )
        self._registry = registry
        self._issuer = issuer
        self._max = max_calls_per_tool
        self._max_tool_calls_per_run = max_tool_calls_per_run
        self._max_distinct_tools_per_run = max_distinct_tools_per_run
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
        # Run-level circuit-breaker state (scalar — the proxy is constructed
        # fresh per pipeline.run(), so it only ever serves one run).
        self._run_total: int = 0
        self._run_tools: set[str] = set()
        self.log: list[dict[str, Any]] = []

    def effects_consumed(self, run_id: str, step: int) -> int:
        """Number of effectful calls the proxy has processed for ``(run_id,
        step)`` — the cursor value used by the pipeline's step-end orphan check."""
        return self._effect_cursor.get((run_id, step), 0)

    def _run_gates(
        self,
        tool: Any,
        tool_ref: str,
        args: Any,
        token: CapabilityToken,
        run_id: str,
        step: Any,
        operation: str,
    ) -> None:
        """Validate and authorise the call (gates 1–5b).

        Gate 0 (registry lookup) is performed by ``invoke`` before this call
        so the resolved tool remains accessible to the ``except ToolError``
        handler regardless of which gate fires. Raises a ``ToolError``
        subclass on any denial; all side-effects (taint ledger, token
        consumption, breaker counts) happen here.
        """
        key = (run_id, step, tool_ref)
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
        # 5b. Run-level circuit-breaker caps (siblings of the per-step
        # breaker; only reached by calls that passed all prior gates).
        # Both predicates are evaluated BEFORE either counter is
        # committed — a breaching call pollutes neither (check-before-
        # commit, intentional asymmetry with the per-step breaker above).
        # Total is checked first (declared ordering), then distinct.
        _is_new_tool = tool_ref not in self._run_tools
        if self._run_total + 1 > self._max_tool_calls_per_run:
            raise CircuitBreakerError(
                f"run '{run_id}' exceeded the max-tool-calls-per-run cap "
                f"(cap {self._max_tool_calls_per_run})"
            )
        if _is_new_tool and len(self._run_tools) + 1 > self._max_distinct_tools_per_run:
            raise CircuitBreakerError(
                f"run '{run_id}' exceeded the max-distinct-tools-per-run cap "
                f"(cap {self._max_distinct_tools_per_run}); "
                f"blocked new tool '{tool_ref}'"
            )
        # Both caps passed — commit run-level state.
        self._run_total += 1
        self._run_tools.add(tool_ref)

    async def _execute_with_ledger(
        self,
        tool: Any,
        args: Any,
        run_id: str,
        step: Any,
        tool_ref: str,
    ) -> "tuple[Any, str | None]":
        """Execute the tool handler, managing the effect ledger when present.

        Returns ``(result, replay_label)`` where ``replay_label`` is
        ``"replay"`` when the result came from a stored effect entry, or
        ``None`` for a live handler call. The caller emits span attributes
        and the log entry for both cases.

        The effect-ledger imports are deferred here for the same import-cycle
        reason as those in ``invoke`` — all modules are fully initialised by
        call time.
        """
        from drawbore.state.effect_ledger import (
            EffectDivergenceError,
            EffectEntry,
            EffectLedgerWriteError,
            EffectStatus,
            EffectUnresolvedError,
            ledger_args_hash,
        )

        from ..access import _reset_idempotency_key, _set_idempotency_key

        # 6. Execute. The effect ledger wraps ONLY the handler call;
        #    every gate above ran first and unchanged, so a denied call
        #    never leaves a phantom pending entry and the taint/token/
        #    breaker guarantees hold identically on resume.
        if self._effect_ledger is None or tool.effectful is False:
            # Read-only / no-ledger path: unchanged, cursor not advanced.
            return await tool.handler(args), None

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
            return result, None
        elif entry.status == EffectStatus.SUCCEEDED and (
            entry.tool_ref,
            entry.input_hash,
        ) == (tool_ref, effect_hash):
            # Replay: return the recorded output with the label; caller emits
            # the span attributes and log entry so it's still observable.
            return entry.output, "replay"
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

        run_id = run_ctx.run_id
        step = getattr(run_ctx, "step", None)
        start = self._clock()
        # Hash the input once and reuse it for both the span tag and the log entry
        # (one identity scheme; avoids divergence between span and log).
        input_hash = payload_hash(args)
        # Initialise to None so the except ToolError handler can detect an
        # unresolved-tool denial (registry.get itself raised ToolAccessError).
        tool = None
        with genai_span(semconv.OP_EXECUTE_TOOL, tool_ref, {
            semconv.GEN_AI_TOOL_NAME: tool_ref,
            semconv.DRAWBORE_RUN_ID: run_id,
            semconv.DRAWBORE_STEP: step,
            semconv.DRAWBORE_TOOL_OPERATION: operation,
            semconv.DRAWBORE_INPUT_HASH: input_hash,
        }) as span:
            try:
                # 0. Resolve the tool here so the except ToolError handler can
                #    access .kind / .exfil_capable for all post-resolution
                #    denials; tool stays None only for an unresolved-tool denial.
                tool = self._registry.get(tool_ref)
                self._run_gates(tool, tool_ref, args, token, run_id, step, operation)
                result, replay_label = await self._execute_with_ledger(
                    tool, args, run_id, step, tool_ref
                )
                if replay_label is not None:
                    # Replay: emit span attributes and log entry, then return.
                    output_hash = payload_hash(result)
                    span.set_attribute(semconv.DRAWBORE_OUTPUT_HASH, output_hash)
                    span.set_attribute(semconv.DRAWBORE_STATUS, replay_label)
                    span.set_status(Status(StatusCode.OK))
                    self._log(
                        tool_ref, run_id, operation, input_hash,
                        output_hash, start, replay_label,
                        step=step,
                        scope=self._ledger.scope(run_id, step).value,
                        kind=tool.kind,
                        exfil_capable=tool.exfil_capable,
                    )
                    return result
            except ToolError as exc:
                label = _denial_label(exc)
                span.set_attribute(semconv.DRAWBORE_STATUS, label)
                self._log(
                    tool_ref, run_id, operation, input_hash, None, start, label,
                    step=step,
                    scope=self._ledger.scope(run_id, step).value,
                    kind=tool.kind if tool is not None else None,
                    exfil_capable=tool.exfil_capable if tool is not None else False,
                )
                raise
            except Exception:
                span.set_attribute(semconv.DRAWBORE_STATUS, "error")
                self._log(
                    tool_ref, run_id, operation, input_hash, None, start, "error",
                    step=step,
                    scope=self._ledger.scope(run_id, step).value,
                    kind=tool.kind if tool is not None else None,
                    exfil_capable=tool.exfil_capable if tool is not None else False,
                )
                raise
            output_hash = payload_hash(result)
            span.set_attribute(semconv.DRAWBORE_OUTPUT_HASH, output_hash)
            span.set_attribute(semconv.DRAWBORE_STATUS, "ok")
            span.set_status(Status(StatusCode.OK))
            self._log(
                tool_ref, run_id, operation, input_hash, output_hash, start, "ok",
                step=step,
                scope=self._ledger.scope(run_id, step).value,
                kind=tool.kind,
                exfil_capable=tool.exfil_capable,
            )
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
        *,
        step: int | None,
        scope: str,
        kind: str | None,
        exfil_capable: bool,
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
                "step": step,
                "scope": scope,
                "kind": kind,
                "exfil_capable": exfil_capable,
            }
        )
