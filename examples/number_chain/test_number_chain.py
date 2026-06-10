"""Provider-free wiring tests for the live OpenAI three-agent chain."""

import importlib.util
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    "number_chain", Path(__file__).with_name("pipeline.py")
)
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


def test_three_agents_are_model_backed_and_chain_previous_output():
    pipeline = example.build_pipeline()

    assert [step.agent.name for step in pipeline.steps] == [
        "draw_number",
        "inspect_number",
        "explain_result",
    ]
    assert [step.agent.spec.model for step in pipeline.steps] == [
        "profile:cheap",
        "profile:judgment",
        "profile:judgment",
    ]
    assert pipeline.steps[1].agent.spec.input is example.DrawnNumber
    assert pipeline.steps[2].agent.spec.input is example.NumberProperties


def test_runtime_config_uses_one_openai_provider():
    config = example.build_llm_config()

    assert set(config.profiles) == {"cheap", "judgment"}
    assert set(config.providers) == {"openai"}
    assert config.providers["openai"].credential_env == "OPENAI_API_KEY"
    assert config.profiles["cheap"].targets[0].provider == "openai"
    assert config.profiles["cheap"].targets[0].model == "gpt-4o-mini"
    assert config.profiles["judgment"].targets[0].provider == "openai"
    assert config.profiles["judgment"].targets[0].model == "gpt-5.4-mini"


def test_copy_paste_command_is_live_openai_command():
    assert "PYTHONPATH=src" in example.COPY_PASTE_COMMAND
    assert "OPENAI_API_KEY=" in example.COPY_PASTE_COMMAND
    assert "examples/number_chain/pipeline.py" in example.COPY_PASTE_COMMAND
