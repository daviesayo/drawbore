from drawbore.llm import CredentialChecker, EnvCredentialChecker


def test_env_checker_reports_present_and_absent(monkeypatch):
    checker = EnvCredentialChecker()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("MISSING_KEY", raising=False)
    assert checker.has_credential(provider="openrouter", credential_env="OPENROUTER_API_KEY") is True
    assert checker.has_credential(provider="openai", credential_env="MISSING_KEY") is False


def test_env_checker_with_no_env_name_cannot_confirm():
    # A providerless/no-env attempt cannot be preflighted: report False (unknown).
    checker = EnvCredentialChecker()
    assert checker.has_credential(provider="x", credential_env=None) is False


def test_env_checker_empty_string_env_var_is_absent(monkeypatch):
    # A set-but-empty key is a real misconfiguration: treat it as unavailable.
    checker = EnvCredentialChecker()
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    assert checker.has_credential(provider="openrouter", credential_env="OPENROUTER_API_KEY") is False


def test_credential_checker_is_a_runtime_protocol():
    class Fake:
        def has_credential(self, *, provider, credential_env):
            return True

    assert isinstance(Fake(), CredentialChecker)  # runtime_checkable Protocol
