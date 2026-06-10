import httpx
import litellm.exceptions as e
import pytest

from drawbore.llm.classify import ProviderOutcome, classify_provider_exception


# Construct provider exceptions the way the verified litellm 1.87.0 API requires.
# Most take (message, model, llm_provider) as keyword args. A few status-bearing
# classes (e.g. PermissionDeniedError) require a non-optional ``response`` — provide
# a fabricated httpx.Response (httpx is a litellm transitive dep, test-only here).
def _mk(cls, **kw):
    resp = httpx.Response(status_code=400, request=httpx.Request("POST", "http://x"))
    try:
        inst = cls(
            message="x", model="gpt-4o", llm_provider="openai", response=resp, **kw
        )
    except TypeError:
        try:
            inst = cls(message="x", model="gpt-4o", llm_provider="openai", **kw)
        except TypeError:
            inst = cls("x")
    # Guard against a future constructor change silently producing the wrong type,
    # which would make the test classify an unintended object.
    assert isinstance(inst, cls), f"_mk produced wrong type for {cls.__name__}"
    return inst


@pytest.mark.parametrize("cls,expected", [
    (e.Timeout, "timeout"),
    (e.RateLimitError, "rate_limit"),
    (e.ServiceUnavailableError, "provider_unavailable"),
    (e.APIConnectionError, "provider_unavailable"),
    (e.InternalServerError, "server_error"),
])
def test_transport_exceptions_classify_to_fallbackable_reasons(cls, expected):
    out = classify_provider_exception(_mk(cls))
    assert out.kind == "transport"
    assert out.reason == expected


@pytest.mark.parametrize("cls", [e.AuthenticationError, e.PermissionDeniedError])
def test_auth_exceptions_classify_as_auth(cls):
    out = classify_provider_exception(_mk(cls))
    assert out.kind == "auth"
    assert out.reason is None


def test_bad_request_classifies_as_contract():
    out = classify_provider_exception(_mk(e.BadRequestError))
    assert out.kind == "contract"


def test_unknown_exception_does_not_become_a_fallback():
    out = classify_provider_exception(ValueError("???"))
    assert out.kind == "unknown"
    assert out.reason is None
