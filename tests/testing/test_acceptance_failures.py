import pytest
from pydantic import BaseModel

from drawbore.agent import agent
from drawbore.pipeline import Pipeline
from drawbore.tools import ToolRegistry


class TxIn(BaseModel):
    transaction_id: str


class Other(BaseModel):
    foo: int


class Fetched(BaseModel):
    transaction_id: str
    amount: int


class Scored(BaseModel):
    amount: int
    risk: str


def _reg():
    reg = ToolRegistry()
    async def real(args):
        raise AssertionError("real handler must not run")
    reg.register_tool("db.read", real, allowed_operations=("invoke",), schema={"type": "object"})
    return reg


def _fetch_agent():
    @agent(name="fetch", input=TxIn, output=Fetched, tools=["db.read"])
    async def fetch(v: TxIn, tools) -> Fetched:
        row = await tools.call("db.read", {"id": v.transaction_id})
        return Fetched(transaction_id=v.transaction_id, amount=row["amount"])
    return fetch


async def test_real_input_schema_violation_halts_before_mocks():   # spec test 2
    p = Pipeline(name="aml", registry=_reg())
    p.add(_fetch_agent())
    async with p.test_mode(mock_tools={"db.read": {"amount": 1}}) as test:
        # Pass a mismatched initial model -> the first step's input validation fails.
        result = await test.run(Other(foo=1))
    assert result.status == "halted"
    assert result.reason.startswith("schema_violation: input:")
    # The audit trace is always built, but no step ran — step_records must be empty.
    # If input validation were bypassed the fetch step would run, call the tool, and
    # record a step entry; a non-empty step_records would falsify this assertion.
    assert result.audit_trace is not None
    assert len(result.audit_trace.step_records) == 0


async def test_real_output_schema_violation_halts_at_the_boundary():   # spec test 3
    @agent(name="score", input=Fetched, output=Scored, model="fake")
    async def score(v: Fetched) -> Scored: ...
    p = Pipeline(name="aml", registry=ToolRegistry())
    p.add(score)
    async with p.test_mode(
        mock_model_responses={"score": {"amount": 1}},   # missing required 'risk'
    ) as test:
        result = await test.run(Fetched(transaction_id="t", amount=1))
    assert result.status == "halted"
    assert result.reason.startswith("schema_violation: output:")


async def test_mocked_tool_runs_through_the_proxy_and_is_audited():   # spec test 4
    p = Pipeline(name="aml", registry=_reg())
    p.add(_fetch_agent())
    async with p.test_mode(mock_tools={"db.read": {"amount": 42}}) as test:
        result = await test.run(TxIn(transaction_id="t1"))
    assert result.status == "completed"
    # the call is recorded in the proxy-derived step audit as an OK invoke
    assert any("db.read (invoke) -> ok" in tc for tc in result.audit_trace.step_records[0].tool_calls)


async def test_operation_denial_is_preserved_for_mocks():   # spec test 5
    reg = ToolRegistry()
    async def real(args):
        return {"amount": 1}
    # original allows only 'read'; the agent calls the default 'invoke' -> denied.
    reg.register_tool("db.read", real, allowed_operations=("read",), schema={"type": "object"})

    @agent(name="fetch", input=TxIn, output=Fetched, tools=["db.read"])
    async def fetch(v: TxIn, tools) -> Fetched:
        row = await tools.call("db.read", {"id": v.transaction_id})  # operation="invoke"
        return Fetched(transaction_id=v.transaction_id, amount=row["amount"])

    p = Pipeline(name="aml", registry=reg)
    p.add(fetch)
    async with p.test_mode(mock_tools={"db.read": {"amount": 1}}) as test:
        result = await test.run(TxIn(transaction_id="t1"))
    assert result.status == "halted"
    assert result.reason.startswith("tool_access")
    # the denial is recorded in the failed step's audit as denied:scope
    assert any("denied:scope" in tc for tc in result.audit_trace.step_records[0].tool_calls)


async def test_circuit_breaker_trips_on_repeated_mocked_calls():   # spec test 6
    reg = ToolRegistry()
    reg.register_tool("db.read", lambda a: {"amount": 1},
                      allowed_operations=("invoke",), schema={"type": "object"})

    @agent(name="fetch", input=TxIn, output=Fetched, tools=["db.read"])
    async def fetch(v: TxIn, tools) -> Fetched:
        for _ in range(10):          # default breaker is 3 calls/tool/step
            await tools.call("db.read", {"id": v.transaction_id})
        return Fetched(transaction_id=v.transaction_id, amount=1)

    p = Pipeline(name="aml", registry=reg)
    p.add(fetch)
    async with p.test_mode(mock_tools={"db.read": {"amount": 1}}) as test:
        result = await test.run(TxIn(transaction_id="t1"))
    assert result.status == "halted"
    assert result.reason.startswith("circuit_breaker")
    assert any("denied:breaker" in tc for tc in result.audit_trace.step_records[0].tool_calls)
