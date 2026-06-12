import json
import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.orchestration import ADKEngine
from drawbore.llm import LLMGateway, ModelResponse
from drawbore.evidence import (
    EvidencePolicy, InMemoryEvidenceStore, EvidenceHandle,
    register_evidence_tool, EVIDENCE_TOOL_REF,
)
from drawbore.tools import ToolRegistry
from drawbore.orchestration.engine import provider_safe_tool_aliases


class Row(BaseModel):
    id: int
    amount: int
    status: str = "ok"


class In(BaseModel):
    records: list[Row]


class Out(BaseModel):
    answer: str


class _Gw(LLMGateway):
    async def complete(self, request):
        return ModelResponse(output={"answer": "x"}, model_used="m", raw_text="{}")


def _seed_handle(store, handle_id="seed-h1"):
    """Pre-seed a retrievable original under a KNOWN handle id (deterministic — the
    model is 'given' this handle via its script, mirroring the loop-mechanism-only
    scope: handle auto-surfacing is deferred, so the developer supplies the id)."""
    original = {"records": [{"id": i, "amount": i * 10} for i in range(200)]}
    store.put(
        EvidenceHandle(
            handle_id=handle_id, run_id="r1", step=0, source_agent="screen",
            content_type="json_rows", original_hash="o" * 16, compressed_hash="c" * 16,
            original_tokens=4000, compressed_tokens=900, transform="json_rows",
        ),
        original=original, compressed={"records": original["records"][:2]},
    )
    store.set_policy(handle_id, allow_full=False, allow_search=True)


async def test_compressed_loop_agent_retrieves_the_original_via_the_proxy_tool(fake_adk_model):
    store = InMemoryEvidenceStore()
    _seed_handle(store)
    reg = ToolRegistry()
    register_evidence_tool(reg, store=store)

    @agent(name="screen", input=In, output=Out, model="fake", tools=[EVIDENCE_TOOL_REF])
    async def screen(v: In) -> Out:
        raise AssertionError("must not run")

    # A real provider calls the tool under its provider-safe alias, not the raw
    # ``evidence://`` ref; the loop resolves the alias back to the canonical ref.
    evidence_alias = provider_safe_tool_aliases((EVIDENCE_TOOL_REF,))[EVIDENCE_TOOL_REF]
    script = [
        ("call", evidence_alias, {"handle_id": "seed-h1", "mode": "search", "query": "990"}),
        ("final", json.dumps({"answer": "screened"})),
    ]
    p = Pipeline(name="aml", registry=reg)
    p.add(screen, evidence=EvidencePolicy(name="aml", enabled=True, min_tokens=100))
    r = await p.run(In(records=[Row(id=i, amount=i * 10) for i in range(400)]),
                    run_id="r1",
                    engine=ADKEngine(gateway=_Gw(), model_factory=lambda n: fake_adk_model(script)),
                    evidence_store=store)
    assert r.status == "completed"
    assert r.outputs["screen"].answer == "screened"
    step = r.audit_trace.step_records[0]
    # compression of THIS step's input happened (evidence compression runs before the loop) ...
    assert step.evidence is not None
    # ... and the evidence tool was a real, proxy-scoped mid-loop call (audited).
    assert any(EVIDENCE_TOOL_REF in tc and "ok" in tc for tc in step.tool_calls)
