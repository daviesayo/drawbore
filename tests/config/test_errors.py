from drawbore.config import ConfigResolutionError
from drawbore.errors import DrawboreError, halt_reason_for


def test_config_resolution_error_is_a_drawbore_error():
    assert issubclass(ConfigResolutionError, DrawboreError)


def test_config_resolution_error_has_a_legible_halt_reason():
    # self-declared halt_reason mechanism.
    assert halt_reason_for(ConfigResolutionError("x")) == "config_resolution_error"


def test_config_resolution_error_carries_its_message():
    assert "agent ref 'pkg:missing' is not registered" in str(
        ConfigResolutionError("agent ref 'pkg:missing' is not registered")
    )
