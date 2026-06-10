"""The remittance pipeline run through the REAL safety layer via test_mode: the
tool and the model are mocked (synthetic data), every control runs for real."""

import importlib.util
from pathlib import Path

from drawbore.escalation import EscalationPolicy

_spec = importlib.util.spec_from_file_location(
    "remittance_example", Path(__file__).with_name("pipeline.py")
)
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)

CANONICAL_TXN = {
    "transaction_id": "txn_demo_1",
    "amount": 250000,
    "currency": "USD",
    "corridor": "US->PH",
    "status": "pending",
}


def _mocks(risk):
    return dict(
        mock_tools={example.TRANSACTION_TOOL: CANONICAL_TXN},
        mock_model_responses={"score_risk": risk},
    )


async def test_low_risk_is_confirmed():
    pipeline = example.build_pipeline(example.build_registry())
    mocks = _mocks({"risk_level": "low", "confidence": 0.95, "factors": ["clean_corridor"]})
    async with pipeline.test_mode(**mocks) as tp:
        result = await tp.run(example.RemittanceRequest(transaction_id="txn_demo_1"))

    assert result.status == "completed"
    confirmation = result.outputs["write_confirmation"]
    assert isinstance(confirmation, example.Confirmation)
    assert confirmation.decision == "confirmed"
    assert "remittance_validation" in result.audit_trace.legible()


async def test_low_confidence_escalates():
    pipeline = example.build_pipeline(
        example.build_registry(),
        on_failure=EscalationPolicy(channel="human_review", target="compliance_ops", mode="sync"),
    )
    mocks = _mocks({"risk_level": "high", "confidence": 0.2, "factors": ["sanctions_corridor"]})
    async with pipeline.test_mode(**mocks) as tp:
        result = await tp.run(example.RemittanceRequest(transaction_id="txn_demo_1"))

    # confidence 0.2 < threshold 0.8 -> the confidence gate escalates
    assert result.status == "escalated"
    assert result.halted_at == "score_risk"
    assert "confidence_below_threshold" in result.reason
    assert len(result.escalations) == 1
    assert "write_confirmation" not in result.outputs
