"""Provider-safe tool-name aliases for the in-step tool loop.

A function-calling provider restricts tool names to ``[A-Za-z0-9_-]`` (OpenAI's
documented set; ADK's own ``FunctionDeclaration.name`` is similarly bounded). A
tool ref like ``mcp://server/tool`` cannot be sent verbatim, so the loop exposes
each tool to the model under a sanitized ALIAS and maps the model's chosen alias
back to the canonical ref before invoking the proxy. Scope enforcement, the JIT
token, and the audit/proxy log all stay keyed on the canonical ref.
"""

import json

import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.orchestration import ADKEngine, ToolLoopBundle
from drawbore.orchestration.engine import provider_safe_tool_aliases
from drawbore.orchestration.scripted_model import make_scripted_model_factory
from drawbore.tools import RunContext, TokenIssuer, ToolProxy, ToolRegistry

MCP_REF = "mcp://sanctions_feed/latest_updates"


class In(BaseModel):
    task: str


class Out(BaseModel):
    answer: str


def _spec():
    @agent(name="feed_reader", input=In, output=Out, model="fake", tools=[MCP_REF])
    async def feed_reader(v: In, tools) -> Out:
        raise AssertionError("model+tools agents run via the loop, not fn")
    return feed_reader.spec


def _bundle():
    reg = ToolRegistry()

    async def latest_updates(args):
        return {"updates": [], "queried": args.get("since")}

    reg.register_tool(
        MCP_REF, latest_updates, allowed_operations=("invoke",),
        schema={"type": "object", "properties": {"since": {"type": "string"}}},
    )
    issuer = TokenIssuer()
    proxy = ToolProxy(reg, issuer)
    return ToolLoopBundle(
        proxy=proxy, issuer=issuer, registry=reg, declared=(MCP_REF,),
        run_ctx=RunContext(run_id="r1", step=0),
    ), proxy


class _RaisingGateway:
    async def complete(self, request):
        raise AssertionError("model+tools loop must not use the gateway")


# --- the alias function itself --------------------------------------------------

def test_alias_is_provider_safe_and_round_trips_to_the_canonical_ref():
    aliases = provider_safe_tool_aliases((MCP_REF,))
    alias = aliases[MCP_REF]
    # provider-safe charset only (OpenAI: a-z A-Z 0-9 _ - ; max 64)
    assert all(c.isalnum() or c in "_-" for c in alias)
    assert "://" not in alias and "/" not in alias and ":" not in alias
    assert 1 <= len(alias) <= 64
    assert (alias[0].isalpha() or alias[0] == "_")


def test_aliases_are_collision_free_after_sanitizing():
    # Two distinct refs that sanitize to the same base must get distinct aliases.
    declared = ("mcp://a/b", "mcp://a_b")  # both -> "mcp___a_b" before disambiguation
    aliases = provider_safe_tool_aliases(declared)
    assert aliases["mcp://a/b"] != aliases["mcp://a_b"]
    assert len(set(aliases.values())) == len(declared)


# --- the real loop resolves an alias back to the canonical ref ------------------

async def test_loop_resolves_alias_to_canonical_ref_through_the_proxy():
    bundle, proxy = _bundle()
    alias = provider_safe_tool_aliases((MCP_REF,))[MCP_REF]
    # A real OpenAI-compatible provider calls the SANITIZED alias name, never the
    # raw ``mcp://`` ref (which the provider could not represent).
    factory = make_scripted_model_factory([
        ("call", alias, {"since": "2026-01-01"}),
        ("text", json.dumps({"answer": "done"})),
    ])
    engine = ADKEngine(gateway=_RaisingGateway(), model_factory=factory, max_llm_calls=8)
    out = await engine.run_step(_spec(), In(task="t"), tool_loop=bundle)
    assert out.output == {"answer": "done"}
    # The proxy/audit log keeps the CANONICAL ref, not the alias — scope enforcement
    # and the audit trail are unchanged.
    ok = [e for e in proxy.log if e["result"] == "ok"]
    assert ok and ok[0]["tool"] == MCP_REF
    assert all(e["tool"] == MCP_REF for e in proxy.log)


async def test_loop_fails_closed_on_an_alias_that_maps_to_no_declared_tool():
    from drawbore.llm import LLMError
    bundle, _proxy = _bundle()
    # The model invents an alias-shaped name that maps to no declared tool.
    factory = make_scripted_model_factory([
        ("call", "totally_invented_tool", {"since": "x"}),
        ("text", json.dumps({"answer": "should not reach"})),
    ])
    engine = ADKEngine(gateway=_RaisingGateway(), model_factory=factory, max_llm_calls=8)
    with pytest.raises(LLMError):
        await engine.run_step(_spec(), In(task="t"), tool_loop=bundle)
