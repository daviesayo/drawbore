"""ADK tool guardrail. Imported only inside ``orchestration``.

A Drawbore tool is exposed to ADK as a ``FunctionTool`` whose callable IS the
proxy-backed invocation — so the tool path is the proxy *by construction*, exactly
as the in-process ``ToolContext`` shim. ``before_tool_callback`` is the second,
independent guardrail: it hard-blocks a tool the agent did not declare (returning a
dict skips the tool in ADK).
"""

from __future__ import annotations

from typing import Any

from google.adk.models import LlmResponse
from google.adk.tools import BaseTool, FunctionTool
from google.genai import types

from drawbore.observability import genai_span, semconv
from drawbore.tools import RunContext, TokenIssuer, ToolProxy


def as_adk_tool(
    tool_ref: str, *, proxy: ToolProxy, issuer: TokenIssuer, run_ctx: RunContext
) -> FunctionTool:
    """Wrap a declared Drawbore tool as an ADK ``FunctionTool`` whose callable
    routes through ``proxy.invoke`` with a single-use JIT token — the SAME
    authoritative path as the in-process tool shim."""

    async def _shim(**kwargs: Any) -> Any:
        # EXACT same authoritative path as the in-process shim in tools/access.py:
        # issue a single-use token, then call proxy.invoke. The model's named
        # arguments arrive as kwargs and become the single args object passed to the
        # proxy. ``run_ctx`` is passed explicitly (not read from the contextvar)
        # because ADK may invoke the tool where the contextvar is not propagated.
        token = issuer.issue(tool_ref, run_ctx.run_id, "invoke")
        return await proxy.invoke(tool_ref, kwargs, token, run_ctx, "invoke")

    _shim.__name__ = tool_ref
    return FunctionTool(func=_shim)


def make_before_tool_callback(declared: tuple[str, ...]):
    """Return an ADK ``before_tool_callback`` that hard-blocks any tool not in the
    agent's declared set (the second, independent guardrail). Returning a dict skips
    the tool; returning ``None`` lets the proxy-backed call proceed."""

    def _callback(tool, args, tool_context) -> dict | None:
        if tool.name not in declared:
            return {"error": f"tool '{tool.name}' is not declared for this agent"}
        return None

    return _callback


_GENAI_TYPE = {
    "string": types.Type.STRING,
    "integer": types.Type.INTEGER,
    "number": types.Type.NUMBER,
    "boolean": types.Type.BOOLEAN,
    "object": types.Type.OBJECT,
    "array": types.Type.ARRAY,
}


def _schema_to_genai(schema: dict | None) -> "types.Schema":
    """Build a genai ``Schema`` from a registry JSON-schema dict so the model sees
    the tool's parameters. Conservative + deterministic: an object with typed
    properties + required. A non-object or absent top-level schema returns an open
    OBJECT schema; an unrecognised property type falls back to STRING — never
    silently-wrong parameters."""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return types.Schema(type=types.Type.OBJECT)
    props = {}
    for key, spec in (schema.get("properties") or {}).items():
        t = (spec or {}).get("type", "string")
        props[key] = types.Schema(type=_GENAI_TYPE.get(t, types.Type.STRING))
    return types.Schema(
        type=types.Type.OBJECT,
        properties=props,
        required=list(schema.get("required") or []),
    )


class _ProxyBackedTool(BaseTool):
    """A declared Drawbore tool exposed to ADK with a registry-schema declaration,
    executed ONLY through ``proxy.invoke`` with a single-use JIT token. On any
    failure it records the exception in ``failures`` (immediate-abort signal) and
    re-raises so ADK also sees the failure."""

    def __init__(self, tool_ref, *, proxy, issuer, run_ctx, schema, failures):
        description = f"Drawbore tool {tool_ref}"
        super().__init__(name=tool_ref, description=description)
        self._ref = tool_ref
        self._description = description
        self._proxy = proxy
        self._issuer = issuer
        self._run_ctx = run_ctx
        self._schema = schema
        self._failures = failures

    def _get_declaration(self):
        return types.FunctionDeclaration(
            name=self._ref,
            description=self._description,
            parameters=_schema_to_genai(self._schema),
        )

    async def run_async(self, *, args, tool_context):
        # EXACT same authoritative path as the in-process shim and ``as_adk_tool``:
        # mint a single-use token, then call proxy.invoke. ``run_ctx`` is passed
        # explicitly because ADK may invoke the tool where the contextvar is not
        # propagated.
        try:
            token = self._issuer.issue(self._ref, self._run_ctx.run_id, "invoke")
            return await self._proxy.invoke(
                self._ref, args, token, self._run_ctx, "invoke"
            )
        except Exception as exc:  # record for immediate-abort, then re-raise
            self._failures.append(exc)
            raise


def proxy_backed_tool(
    tool_ref: str, *, proxy: ToolProxy, issuer: TokenIssuer, run_ctx: RunContext,
    schema: dict | None, failures: list,
) -> BaseTool:
    """Build the proxy-backed, schema-backed ADK tool for one declared ref.

    The tool's declaration is derived from the registry tool's JSON ``schema`` so
    the model sees its parameters; execution routes through ``proxy.invoke`` with a
    single-use JIT token; any failure is recorded in ``failures`` (immediate-abort
    signal) and re-raised so ADK also sees it."""
    return _ProxyBackedTool(
        tool_ref, proxy=proxy, issuer=issuer, run_ctx=run_ctx,
        schema=schema, failures=failures,
    )


def _terminating_content() -> "types.Content":
    """A minimal, content-only model turn used as the short-circuit response. Its
    text is irrelevant — the engine re-raises the real recorded failure later; this
    only needs to be a valid ``LlmResponse`` body so ADK treats the model call as
    handled (skipped)."""
    return types.Content(role="model", parts=[types.Part(text="{}")])


def make_loop_model_callbacks(bundle, *, run_id: str):
    """Return ``(before_model_callback, after_model_callback)`` for the loop. Each
    genuinely-started model turn is counted in ``bundle.turns`` (``before_model``)
    and emitted as a ``chat`` span nested under the active ``invoke_agent`` span
    when the turn completes (``after_model``). The instant a tool failure is
    recorded, the before callback returns a terminating ``LlmResponse`` so ADK
    computes no further model turn (the secondary guard behind the engine hard-break).

    The span is opened and closed entirely inside ``after_model`` via a single
    ``with`` block, so it can never leak — ``before_model``/``after_model`` do NOT
    straddle one context manager (if ADK errored mid-turn and skipped ``after_model``
    a straddled span would never close, corrupting the trace hierarchy; the
    one-shot path uses the same single-``with`` discipline). A ``pending`` marker
    started in ``before_model`` and consumed in ``after_model`` ensures a
    short-circuited (failure) turn — which counts no turn — emits no span even if ADK
    still calls ``after_model`` on the synthesised response.

    Signatures match ADK's ``LlmAgent`` callbacks: ``before_model(callback_context,
    llm_request) -> LlmResponse | None`` (an ``LlmResponse`` SKIPS the model call)
    and ``after_model(callback_context, llm_response) -> LlmResponse | None``."""
    pending: list = []

    def before_model(callback_context, llm_request):
        if bundle.failures:
            return LlmResponse(content=_terminating_content())
        bundle.turns.append(1)  # one marker per model turn; count == len(turns)
        pending.append(1)       # this turn is owed a span when it completes
        return None

    def after_model(callback_context, llm_response):
        if pending:
            pending.pop()
            # The loop-path chat span does NOT carry drawbore.declared_model — the
            # declared-ref tag is set on the one-shot path only (adk_engine.py).
            # This callback sees only the resolved model name, not the declared ref.
            with genai_span(
                semconv.OP_CHAT,
                "loop",
                {
                    semconv.DRAWBORE_RUN_ID: run_id,
                    semconv.GEN_AI_OPERATION_NAME: semconv.OP_CHAT,
                },
            ):
                pass            # the model work already happened; the span marks it
        return None

    return before_model, after_model


def make_loop_before_tool_callback(bundle):
    """Return a ``before_tool_callback`` that blocks any tool not in the agent's
    declared set AND blocks any tool once a failure is recorded (so no further tool
    runs after the failed control). A dict return skips the tool in ADK.

    Signature matches ADK: ``before_tool(tool, args, tool_context) -> dict | None``.
    The undeclared-tool guard is delegated to ``make_before_tool_callback`` (single
    source of truth); the loop variant adds the after-failure block and, when a
    declared tool is about to run, marks ``bundle.tool_invoked`` so the loop
    driver can tell a pre-tool provider failure (may fall back) from a post-tool one
    (must fail closed). A DENIED tool (undeclared, or after a prior failure) does NOT
    mark ``tool_invoked`` — only a tool that actually proceeds counts as having run.
    """
    undeclared_guard = make_before_tool_callback(bundle.declared)

    def before_tool(tool, args, tool_context):
        if bundle.failures:
            return {"error": "tool loop aborted after a prior tool failure"}
        blocked = undeclared_guard(tool, args, tool_context)
        if blocked is not None:
            return blocked  # undeclared/denied — does NOT count as a tool invocation
        # The tool is declared and the loop is healthy: it is about to run. Mark it so a
        # subsequent provider failure is treated as post-tool (fail closed, never replay).
        bundle.tool_invoked.append(1)
        return None

    return before_tool
