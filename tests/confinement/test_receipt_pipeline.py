"""Integration tests: confinement receipt minted on every run via pipeline.test_mode.

Verifies that:
- A clean run with a tool call produces a "confined" receipt whose observed_calls
  match the proxy log.
- A denied:taint call is recorded in the receipt and the verdict is still "confined"
  (a denial is evidence a control fired, not a breach).
- A halted run (schema violation) still carries a confinement_receipt in its finally
  block regardless of the exit path.
- receipt.legible() returns a non-empty readable string.
- Two runs of the same pipeline with the same run_id and inputs produce the same
  receipt_fingerprint (determinism).
"""

from __future__ import annotations

from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.confinement import verify
from drawbore.pipeline.binding import From
from drawbore.tools import ToolRegistry
from drawbore.testing import call, final


# ---------------------------------------------------------------------------
# Shared schemas
# ---------------------------------------------------------------------------


class SimpleIn(BaseModel):
    x: int


class SimpleOut(BaseModel):
    y: int


class Mid(BaseModel):
    note: str


class FinalOut(BaseModel):
    ok: bool


# ---------------------------------------------------------------------------
# Test 1: Clean run — receipt is confined, observed_calls reflect tool calls
# ---------------------------------------------------------------------------


TOOL_REF = "conf:clean"


async def test_clean_run_receipt_is_confined():
    """A single-step model+tools agent that calls one trusted tool: the receipt
    is 'confined', observed_calls has one entry with result='ok', and verify()
    agrees."""
    async def _handler(args):
        return {"y": 10}

    reg = ToolRegistry()
    reg.register_tool(TOOL_REF, _handler)

    @agent(name="rp_proc", input=SimpleIn, output=SimpleOut, model="m-1.0", tools=[TOOL_REF])
    async def rp_proc(v: SimpleIn) -> SimpleOut: ...

    pipe = Pipeline("rp-clean", registry=reg).add(rp_proc)
    async with pipe.test_mode(
        mock_loop_scripts={"rp_proc": [call(TOOL_REF), final({"y": 10})]},
        mock_tools={TOOL_REF: {"y": 10}},
    ) as tp:
        result = await tp.run(SimpleIn(x=1), run_id="rp-clean-1")

    assert result.status == "completed"
    assert result.confinement_receipt is not None

    receipt = result.confinement_receipt
    # The verdict should be "confined": one ok call to a declared tool
    verdict = verify(receipt)
    assert verdict.status == "confined", (
        f"expected confined, got {verdict.status}: {verdict.breaches} / {verdict.unverifiable}"
    )
    # observed_calls reflects the single tool invocation
    assert len(receipt.observed_calls) == 1
    oc = receipt.observed_calls[0]
    assert oc.tool_ref == TOOL_REF
    assert oc.result == "ok"
    assert oc.agent == "rp_proc"


# ---------------------------------------------------------------------------
# Test 2: Taint denied:taint — receipt records the denial, verdict is confined
# ---------------------------------------------------------------------------


SRC_REF = "conf:src"
SINK_REF = "conf:sink"


async def test_taint_denied_call_recorded_and_verdict_confined():
    """Two-step pipeline with explicit bindings: step 1 calls an UNTRUSTED-source
    tool (taints scope), step 2 tries to call an exfil-capable sink which is
    denied:taint. The run halts with taint_violation, but the receipt is still
    minted (finally block), records the denial, and the verdict is 'confined'
    because a fired control is not a breach."""
    async def _h(args):
        return {"ok": True}

    reg = ToolRegistry()
    reg.register_mcp_tool(SRC_REF, _h)                      # UNTRUSTED source (MCP default)
    reg.register_tool(SINK_REF, _h, exfil_capable=True)     # exfil-capable sink

    @agent(name="rp_fetcher", input=SimpleIn, output=Mid, model="m-1.0", tools=[SRC_REF])
    async def rp_fetcher(v: SimpleIn) -> Mid: ...

    @agent(name="rp_exporter", input=Mid, output=FinalOut, model="m-1.0", tools=[SINK_REF])
    async def rp_exporter(v: Mid) -> FinalOut: ...

    pipe = (
        Pipeline("rp-taint", registry=reg)
        .add(rp_fetcher)
        .add(
            rp_exporter,
            inputs={"note": From("rp_fetcher.note")},
            depends_on=["rp_fetcher"],
        )
    )
    async with pipe.test_mode(
        mock_loop_scripts={
            "rp_fetcher": [call(SRC_REF), final({"note": "data"})],
            "rp_exporter": [call(SINK_REF), final({"ok": True})],
        },
        mock_tools={SRC_REF: {"note": "data"}, SINK_REF: {"ok": True}},
    ) as tp:
        result = await tp.run(SimpleIn(x=1), run_id="rp-taint-1")

    # The pipeline halts on taint_violation
    assert result.status in ("halted", "escalated")
    assert result.reason is not None and "taint" in result.reason

    # The receipt is still minted in the finally block
    assert result.confinement_receipt is not None
    receipt = result.confinement_receipt

    # The denied:taint call must appear in observed_calls
    denied_calls = [c for c in receipt.observed_calls if "denied:taint" in c.result]
    assert denied_calls, (
        f"expected a denied:taint call; observed_calls: {receipt.observed_calls}"
    )

    # A denial is not a breach — the verdict must be 'confined'
    verdict = verify(receipt)
    assert verdict.status == "confined", (
        f"expected confined, got {verdict.status}: "
        f"breaches={verdict.breaches} / unverifiable={verdict.unverifiable}"
    )


# ---------------------------------------------------------------------------
# Test 3: Halted run (schema violation) still carries a receipt
# ---------------------------------------------------------------------------


async def test_halted_run_still_has_receipt():
    """A run that halts on schema_violation still has a confinement_receipt on the
    RunResult — the finally block mints it regardless of the exit path."""

    class BadOut(BaseModel):
        must_exist: int

    @agent(name="rp_bad_agent", input=SimpleIn, output=BadOut, model="m-1.0")
    async def rp_bad_agent(v: SimpleIn) -> BadOut: ...

    pipe = Pipeline("rp-halt").add(rp_bad_agent)
    async with pipe.test_mode(
        mock_model_responses={"rp_bad_agent": {"wrong_field": "not an int"}},
    ) as tp:
        result = await tp.run(SimpleIn(x=1), run_id="rp-halt-1")

    assert result.status == "halted"
    assert result.reason is not None and "schema_violation" in result.reason

    # Receipt is minted even on a halt
    assert result.confinement_receipt is not None
    receipt = result.confinement_receipt
    assert receipt.run_id == "rp-halt-1"
    assert receipt.status == "halted"


# ---------------------------------------------------------------------------
# Test 4: legible() returns a non-empty readable string
# ---------------------------------------------------------------------------


async def test_receipt_legible_is_non_empty():
    """A clean deterministic run's receipt.legible() returns a non-empty string
    containing the run_id and the 'CONFINED' verdict."""

    @agent(name="rp_readable", input=SimpleIn, output=SimpleOut)
    async def rp_readable(v: SimpleIn) -> SimpleOut:
        return SimpleOut(y=v.x * 2)

    pipe = Pipeline("rp-legible").add(rp_readable)
    async with pipe.test_mode() as tp:
        result = await tp.run(SimpleIn(x=3), run_id="rp-legible-1")

    assert result.status == "completed"
    assert result.confinement_receipt is not None

    text = result.confinement_receipt.legible()
    assert isinstance(text, str)
    assert len(text) > 0
    assert "rp-legible-1" in text
    assert "CONFINED" in text.upper()


# ---------------------------------------------------------------------------
# Test 5: Determinism — same pipeline + inputs + run_id → same fingerprint
# ---------------------------------------------------------------------------


async def test_determinism_same_inputs_same_fingerprint():
    """Two independent runs of the same deterministic pipeline with the same
    run_id and inputs produce identical receipt_fingerprints."""

    @agent(name="rp_det", input=SimpleIn, output=SimpleOut)
    async def rp_det(v: SimpleIn) -> SimpleOut:
        return SimpleOut(y=v.x + 1)

    pipe = Pipeline("rp-det").add(rp_det)

    async with pipe.test_mode() as tp:
        r1 = await tp.run(SimpleIn(x=5), run_id="rp-det-1")

    async with pipe.test_mode() as tp:
        r2 = await tp.run(SimpleIn(x=5), run_id="rp-det-1")

    assert r1.confinement_receipt is not None
    assert r2.confinement_receipt is not None
    assert r1.confinement_receipt.receipt_fingerprint == r2.confinement_receipt.receipt_fingerprint
    assert r1.confinement_receipt.receipt_fingerprint.startswith("sha256:")
