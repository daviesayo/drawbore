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
from google.adk.tools import BaseTool
from google.genai import types

from drawbore.observability import genai_span, semconv
from drawbore.tools import RunContext, TokenIssuer, ToolProxy, authorized_invoke

from .engine import provider_safe_tool_aliases



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
    re-raises so ADK also sees the failure.

    The tool is exposed to the model under a provider-safe ALIAS ``name`` (a function
    name a provider can represent), while every authoritative action — token issuance,
    ``proxy.invoke``, the proxy/audit log — uses the canonical ``tool_ref``. The model
    selects the alias; Drawbore resolves it back to the canonical ref by construction
    (the callable closes over ``tool_ref``), so scope enforcement and audit are
    unchanged."""

    def __init__(self, tool_ref, *, name, proxy, issuer, run_ctx, schema, failures):
        description = f"Drawbore tool {tool_ref}"
        super().__init__(name=name, description=description)
        self._ref = tool_ref
        self._description = description
        self._proxy = proxy
        self._issuer = issuer
        self._run_ctx = run_ctx
        self._schema = schema
        self._failures = failures

    def _get_declaration(self):
        return types.FunctionDeclaration(
            name=self.name,
            description=self._description,
            parameters=_schema_to_genai(self._schema),
        )

    async def run_async(self, *, args, tool_context):
        # Same authoritative path as the in-process shim: ``run_ctx`` is passed
        # explicitly because ADK may invoke the tool where the contextvar is not
        # propagated.
        try:
            return await authorized_invoke(
                self._ref, args, "invoke", self._proxy, self._issuer, self._run_ctx
            )
        except Exception as exc:  # record for immediate-abort, then re-raise
            self._failures.append(exc)
            raise


def proxy_backed_tool(
    tool_ref: str, *, name: str | None = None, proxy: ToolProxy, issuer: TokenIssuer,
    run_ctx: RunContext, schema: dict | None, failures: list,
) -> BaseTool:
    """Build the proxy-backed, schema-backed ADK tool for one declared ref.

    The tool's declaration is derived from the registry tool's JSON ``schema`` so
    the model sees its parameters; the tool is exposed under the provider-safe
    ``name`` (defaulting to the sanitized alias of ``tool_ref`` when omitted), while
    execution routes through ``proxy.invoke`` on the canonical ``tool_ref`` with a
    single-use JIT token; any failure is recorded in ``failures`` (immediate-abort
    signal) and re-raised so ADK also sees it."""
    exposed = name if name is not None else provider_safe_tool_aliases((tool_ref,))[tool_ref]
    return _ProxyBackedTool(
        tool_ref, name=exposed, proxy=proxy, issuer=issuer, run_ctx=run_ctx,
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
    pending_turn: bool = False

    def before_model(callback_context, llm_request):
        nonlocal pending_turn
        if bundle.failures:
            return LlmResponse(content=_terminating_content())
        bundle.turns.append(1)  # one marker per model turn; count == len(turns)
        pending_turn = True     # this turn is owed a span when it completes
        return None

    def after_model(callback_context, llm_response):
        nonlocal pending_turn
        if pending_turn:
            pending_turn = False
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


def make_loop_before_tool_callback(bundle, *, exposed_names: tuple):
    """Return a ``before_tool_callback`` that blocks any tool not in the agent's
    declared set AND blocks any tool once a failure is recorded (so no further tool
    runs after the failed control). A dict return skips the tool in ADK.

    Signature matches ADK: ``before_tool(tool, args, tool_context) -> dict | None``.
    When a declared tool is about to run, marks ``bundle.tool_invoked`` so the loop
    driver can tell a pre-tool provider failure (may fall back) from a post-tool one
    (must fail closed). A DENIED tool (undeclared, or after a prior failure) does NOT
    mark ``tool_invoked`` — only a tool that actually proceeds counts as having run.

    The model sees each tool under its provider-safe ALIAS, so the undeclared guard is
    keyed on ``exposed_names`` — the caller passes the aliases it already computed for
    the tool builder, making the agreement structural. A model-invented alias that maps
    to no declared tool is blocked here as undeclared (and ADK, finding no matching
    tool, also fails the call); either way the loop halts and never reaches the proxy."""

    def before_tool(tool, args, tool_context):
        if bundle.failures:
            return {"error": "tool loop aborted after a prior tool failure"}
        if tool.name not in exposed_names:
            return {"error": f"tool '{tool.name}' is not declared for this agent"}
        # The tool is declared and the loop is healthy: it is about to run. Mark it so a
        # subsequent provider failure is treated as post-tool (fail closed, never replay).
        bundle.tool_invoked.append(1)
        return None

    return before_tool
