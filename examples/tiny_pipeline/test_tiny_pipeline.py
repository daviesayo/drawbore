"""The tiny pipeline runs through the REAL safety layer via test_mode (no mocks
needed — it is fully deterministic)."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "tiny_pipeline_example", Path(__file__).with_name("pipeline.py")
)
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


async def test_tiny_pipeline_completes():
    pipeline = example.build_pipeline()
    async with pipeline.test_mode() as tp:
        result = await tp.run(example.Payment(cents=2500))

    assert result.status == "completed"
    categorized = result.outputs["categorize"]
    assert isinstance(categorized, example.Categorized)
    assert categorized.dollars == 25.0
    assert categorized.tier == "small"
    assert "tiny_payments" in result.audit_trace.legible()
