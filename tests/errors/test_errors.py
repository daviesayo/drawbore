from drawbore.errors import DrawboreError, SanitizationError, halt_reason_for
from drawbore.schema.errors import SchemaValidationError
from drawbore.tools.errors import CircuitBreakerError, TokenError, ToolAccessError


def test_sanitization_error_is_a_drawbore_error():
    assert issubclass(SanitizationError, DrawboreError)


def test_halt_reason_classifies_known_failures():
    assert halt_reason_for(SchemaValidationError(int, [])) == "schema_violation"
    assert halt_reason_for(CircuitBreakerError("x")) == "circuit_breaker"
    assert halt_reason_for(TokenError("x")) == "token_violation"
    assert halt_reason_for(ToolAccessError("x")) == "tool_access"
    assert halt_reason_for(SanitizationError("x")) == "sanitization"


def test_halt_reason_defaults_to_agent_error():
    assert halt_reason_for(RuntimeError("boom")) == "agent_error"
    assert halt_reason_for(ValueError("nope")) == "agent_error"


def test_plain_drawbore_error_defaults_to_agent_error():
    assert halt_reason_for(DrawboreError("x")) == "agent_error"


def test_halt_reason_honours_self_declared_reason_for_layering_isolated_errors():
    # ``llm`` errors cannot be listed in ``_HALT_REASONS`` (errors.py must not
    # import llm — that would be a cycle), so they self-declare a legible
    # ``halt_reason`` class attribute that ``halt_reason_for`` honours.
    from drawbore.llm import LLMError, ModelUnavailableError

    assert halt_reason_for(ModelUnavailableError("x")) == "model_unavailable"
    assert halt_reason_for(LLMError("y")) == "model_error"
