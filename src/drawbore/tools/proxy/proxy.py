"""The tool proxy — the single chokepoint for every tool call.

No tool call reaches its handler except through ``ToolProxy.invoke``. The proxy
validates the single-use capability token, enforces tool-level scope, logs the
call (input/output hashes, duration), and trips a per-agent-step circuit breaker.
"""

from __future__ import annotations

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
    ):
        self._registry = registry
        self._issuer = issuer
        self._max = max_calls_per_tool
        self._clock = clock
        self._ledger = ledger if ledger is not None else TaintLedger()
        # Circuit breaker is per (run, agent-step, tool) — audit H1.
        self._counts: dict[tuple[str, Any, str], int] = {}
        self.log: list[dict[str, Any]] = []

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
                # 6. Execute.
                result = await tool.handler(args)
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
