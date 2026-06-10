import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.orchestration import ADKEngine
from drawbore.llm import LLMGateway, ModelResponse
from drawbore.evidence import EvidencePolicy, InMemoryEvidenceStore
from drawbore.observability import semconv


class Row(BaseModel):
    id: int
    amount: int
    status: str = "ok"


class In(BaseModel):
    records: list[Row]


class Out(BaseModel):
    seen: int


class _CapturingGateway(LLMGateway):
    """Records how many rows the model actually received in its user message."""

    def __init__(self):
        self.last_user = None

    async def complete(self, request):
        self.last_user = request.user
        return ModelResponse(output={"seen": request.user.count('"id"')}, model_used="m", raw_text="{}")


def _records(n):
    return In(records=[Row(id=i, amount=i * 10) for i in range(n)])


async def test_disabled_policy_model_sees_everything(captured_spans):
    gw = _CapturingGateway()

    @agent(name="screen", input=In, output=Out, model="m")
    async def screen(v: In) -> Out:  # body not called on the model path
        raise AssertionError("model-backed fn must not run")

    p = Pipeline(name="aml")
    p.add(screen)  # no evidence= → passthrough
    r = await p.run(_records(300), engine=ADKEngine(gateway=gw))
    assert r.status == "completed"
    assert gw.last_user.count('"id"') == 300   # model saw all rows


async def test_enabled_policy_compresses_the_model_view_and_stores_original(captured_spans):
    gw = _CapturingGateway()
    store = InMemoryEvidenceStore()

    @agent(name="screen", input=In, output=Out, model="m")
    async def screen(v: In) -> Out:
        raise AssertionError("model-backed fn must not run")

    p = Pipeline(name="aml")
    p.add(screen, evidence=EvidencePolicy(name="aml", enabled=True, min_tokens=100))
    r = await p.run(_records(500), engine=ADKEngine(gateway=gw), evidence_store=store)
    assert r.status == "completed"
    assert gw.last_user.count('"id"') < 500    # model saw a bounded view
    step = r.audit_trace.step_records[0]
    assert step.evidence is not None and "compressed" in step.evidence
    span = [s for s in captured_spans.get_finished_spans() if s.name == "invoke_agent screen"][0]
    assert span.attributes[semconv.DRAWBORE_EVIDENCE_DECISION] == "compressed"
    assert span.attributes[semconv.DRAWBORE_EVIDENCE_TRANSFORM] == "json_rows"


async def test_deterministic_agent_is_never_compressed():
    @agent(name="plain", input=In, output=Out)
    async def plain(v: In) -> Out:
        return Out(seen=len(v.records))

    p = Pipeline(name="aml")
    p.add(plain, evidence=EvidencePolicy(enabled=True, min_tokens=1))
    r = await p.run(_records(500))   # LocalEngine, deterministic
    assert r.status == "completed"
    assert r.outputs["plain"].seen == 500          # full data reached the agent
    assert r.audit_trace.step_records[0].evidence is None


async def test_simulate_records_decision_but_model_sees_everything(captured_spans):
    gw = _CapturingGateway()

    @agent(name="screen", input=In, output=Out, model="m")
    async def screen(v: In) -> Out:
        raise AssertionError("model-backed fn must not run")

    p = Pipeline(name="aml")
    p.add(screen, evidence=EvidencePolicy(enabled=True, mode="simulate", min_tokens=100))
    r = await p.run(_records(500), engine=ADKEngine(gateway=gw), evidence_store=InMemoryEvidenceStore())
    assert gw.last_user.count('"id"') == 500       # unchanged payload
    assert "simulate" in r.audit_trace.step_records[0].evidence


async def test_nonstrict_revalidation_failure_passes_through_and_leaves_no_orphan(captured_spans):
    # If the compressed view violates the input model's constraints and the policy
    # is non-strict, the model must receive the FULL original (never an invalid view),
    # the audit must record passthrough, and the store must hold NO orphaned entry.
    from pydantic import conlist

    class InMin(BaseModel):
        # The compressed view (a handful of rows) will violate this min_length.
        records: conlist(Row, min_length=400)

    gw = _CapturingGateway()
    store = InMemoryEvidenceStore()

    @agent(name="screen", input=InMin, output=Out, model="m")
    async def screen(v: InMin) -> Out:
        raise AssertionError("model-backed fn must not run")

    p = Pipeline(name="aml")
    p.add(screen, evidence=EvidencePolicy(enabled=True, min_tokens=100, strict=False))
    r = await p.run(InMin(records=[Row(id=i, amount=i) for i in range(500)]),
                    engine=ADKEngine(gateway=gw), evidence_store=store)
    assert r.status == "completed"
    assert gw.last_user.count('"id"') == 500                  # full original reached the model
    assert r.audit_trace.step_records[0].evidence is not None
    assert "passthrough" in r.audit_trace.step_records[0].evidence
    assert store._entries == {}                               # no orphaned compressed entry


async def test_strict_revalidation_failure_halts_and_model_is_never_called(captured_spans):
    # strict=True: a compressed view that fails re-validation HALTS — the model is
    # never sent anything (not the invalid view, not the full original).
    from pydantic import conlist

    class InMin(BaseModel):
        records: conlist(Row, min_length=400)

    gw = _CapturingGateway()

    @agent(name="screen", input=InMin, output=Out, model="m")
    async def screen(v: InMin) -> Out:
        raise AssertionError("model-backed fn must not run")

    p = Pipeline(name="aml")
    p.add(screen, evidence=EvidencePolicy(enabled=True, min_tokens=100, strict=True))
    r = await p.run(InMin(records=[Row(id=i, amount=i) for i in range(500)]),
                    engine=ADKEngine(gateway=gw), evidence_store=InMemoryEvidenceStore())
    assert r.status == "halted"
    assert "evidence_error" in r.audit_trace.reason
    assert gw.last_user is None                       # the model was never called


async def test_store_failure_halts_and_escalates_with_evidence_reason():
    gw = _CapturingGateway()

    class _BrokenStore(InMemoryEvidenceStore):
        def put(self, *a, **k):
            from drawbore.evidence import EvidenceStoreError
            raise EvidenceStoreError("disk full")

    @agent(name="screen", input=In, output=Out, model="m")
    async def screen(v: In) -> Out:
        raise AssertionError("model-backed fn must not run")

    p = Pipeline(name="aml")
    p.add(screen, evidence=EvidencePolicy(enabled=True, min_tokens=100, require_original_store=True))
    r = await p.run(_records(500), engine=ADKEngine(gateway=gw), evidence_store=_BrokenStore())
    assert r.status == "halted"
    assert "evidence_error" in r.audit_trace.reason
