"""Smoke-example guard: verifies the live OpenAI example wiring without calling
OpenAI. The example itself performs the provider call when run as a script."""

import importlib.util
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    "live_openai_smoke", Path(__file__).with_name("pipeline.py")
)
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


def test_runtime_config_uses_one_openai_provider():
    config = example.build_llm_config()

    assert set(config.profiles) == {"judgment"}
    assert set(config.providers) == {"openai"}
    assert config.providers["openai"].credential_env == "OPENAI_API_KEY"
    assert config.profiles["judgment"].targets[0].provider == "openai"
    assert config.profiles["judgment"].targets[0].model == "gpt-4o-mini"


def test_pipeline_declares_runtime_profile_not_provider_detail():
    pipeline = example.build_pipeline()
    spec = pipeline.steps[0].agent.spec

    assert spec.model == "profile:judgment"
    assert spec.fallback_model is None


def test_copy_paste_command_mentions_live_key_and_example_path():
    assert "PYTHONPATH=src" in example.COPY_PASTE_COMMAND
    assert "OPENAI_API_KEY=" in example.COPY_PASTE_COMMAND
    assert "examples/live_openai_smoke/pipeline.py" in example.COPY_PASTE_COMMAND
