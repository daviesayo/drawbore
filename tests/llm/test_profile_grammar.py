import pytest

from drawbore.errors import halt_reason_for
from drawbore.llm import LLMConfigError
from drawbore.llm.resolution import is_profile_ref, parse_profile_ref


def test_llm_config_error_classifies_as_model_config_error():
    assert halt_reason_for(LLMConfigError("bad setup")) == "model_config_error"


def test_valid_profile_ref_parses_to_name():
    assert is_profile_ref("profile:judgment") is True
    assert parse_profile_ref("profile:judgment") == "judgment"
    assert parse_profile_ref("profile:judgment_backup-2.0") == "judgment_backup-2.0"


def test_direct_strings_are_not_profile_refs():
    assert is_profile_ref("openrouter/anthropic/claude-3-5-sonnet") is False
    assert is_profile_ref("gpt-4o") is False


@pytest.mark.parametrize("bad", [
    "profile:",                # empty name
    "profile: judgment",       # whitespace
    "profile:judgment ",       # trailing whitespace
    "profile:profile:judgment",# nested prefix
    "profile:1bad",            # must start with a letter
    "profile:has/slash",       # invalid char
])
def test_malformed_profile_refs_raise_config_error(bad):
    assert is_profile_ref(bad) is True            # it claims the prefix...
    with pytest.raises(LLMConfigError):           # ...but is malformed
        parse_profile_ref(bad)
