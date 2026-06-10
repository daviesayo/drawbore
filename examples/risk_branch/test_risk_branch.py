"""Branch + join example tests: verifies the risk pipeline runs deterministically
with both the high-risk and low-risk paths, checking that the skipped branch
appears in the audit trace."""

import importlib.util
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    "risk_branch", Path(__file__).with_name("pipeline.py")
)
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


async def test_high_risk_takes_enhanced_branch():
    r = await example.build().run(example.Txn(amount=50000))
    assert r.status == "completed"
    assert r.outputs["writer"].verdict == "enhanced"
    skipped = {s.agent for s in r.audit_trace.step_records if s.status == "skipped"}
    assert "standard_review" in skipped


async def test_low_risk_takes_standard_branch():
    r = await example.build().run(example.Txn(amount=100))
    assert r.status == "completed"
    assert r.outputs["writer"].verdict == "standard"
    skipped = {s.agent for s in r.audit_trace.step_records if s.status == "skipped"}
    assert "enhanced_dd" in skipped


async def test_skipped_branch_explained_in_audit():
    r = await example.build().run(example.Txn(amount=100))
    skip_records = [s for s in r.audit_trace.step_records if s.status == "skipped"]
    names = {s.agent for s in skip_records}
    assert "enhanced_dd" in names
    enh = next(s for s in skip_records if s.agent == "enhanced_dd")
    assert enh.condition is not None
    assert "risk.level" in enh.condition
