from drawbore.llm import ModelAttemptAudit, ModelAudit


def test_model_audit_legible_has_no_secret_values():
    # The audit carries env NAMES at most, never secret values; legible() never
    # renders credential_env or a key. Build an audit and assert no secret leaks.
    audit = ModelAudit(
        declared_refs=("profile:judgment",), selected_provider="openrouter",
        selected_model="anthropic/claude-3-5-sonnet",
        attempts=(ModelAttemptAudit(
            index=0, declared_ref="profile:judgment", source="profile",
            provider="openrouter", model="anthropic/claude-3-5-sonnet",
            request_model="openrouter/anthropic/claude-3-5-sonnet", outcome="success",
        ),),
    )
    text = audit.legible()
    assert "API_KEY" not in text          # credential_env / key material is never rendered by legible()
    assert "sk-" not in text              # no key material
