from drawbore.llm import ModelAttemptAudit, ModelAudit


def test_attempt_audit_fields():
    a = ModelAttemptAudit(
        index=0, declared_ref="profile:judgment", source="profile",
        provider="openrouter", model="anthropic/claude-3-5-sonnet",
        request_model="openrouter/anthropic/claude-3-5-sonnet",
        outcome="success",
    )
    assert a.reason is None and a.outcome == "success"


def test_model_audit_legible_lists_declared_and_attempts():
    audit = ModelAudit(
        declared_refs=("profile:judgment",),
        selected_provider="openai", selected_model="gpt-4o",
        attempts=(
            ModelAttemptAudit(
                index=0, declared_ref="profile:judgment", source="profile",
                provider="openrouter", model="anthropic/claude-3-5-sonnet",
                request_model="openrouter/anthropic/claude-3-5-sonnet",
                outcome="fallback", reason="timeout",
            ),
            ModelAttemptAudit(
                index=1, declared_ref="profile:judgment", source="profile",
                provider="openai", model="gpt-4o", request_model="openai/gpt-4o",
                outcome="success",
            ),
        ),
    )
    text = audit.legible()
    assert "profile:judgment" in text
    assert "openrouter/anthropic/claude-3-5-sonnet -> fallback (timeout)" in text
    assert "openai/gpt-4o -> success" in text
    assert "selected openai/gpt-4o" in text


def test_default_loop_fallback_phase_is_not_loop():
    audit = ModelAudit(declared_refs=("gpt-4o",), selected_provider=None,
                       selected_model="gpt-4o", attempts=())
    assert audit.loop_fallback_phase == "not_loop"
    # providerless bare model renders without a prefix and omits the loop phase
    text = audit.legible()
    assert "selected gpt-4o" in text
    assert "loop fallback" not in text
