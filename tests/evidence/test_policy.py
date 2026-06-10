from drawbore.evidence import EvidencePolicy


def test_policy_defaults_are_opt_in_and_conservative():
    p = EvidencePolicy()
    assert p.enabled is False
    assert p.mode == "compress"
    assert p.allowed_transforms == ("json_rows", "logs")
    assert p.require_original_store is True
    assert p.allow_full_retrieval is False
    assert p.allow_search_retrieval is True
    assert p.strict is False


def test_policy_is_validated():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        EvidencePolicy(mode="nonsense")


def test_policy_label_is_stable_for_audit():
    p = EvidencePolicy(name="aml", enabled=True)
    assert p.name == "aml"


def test_allowed_transforms_coerces_to_a_tuple():
    # Pydantic v2 coerces a list to the tuple field; callers can rely on a tuple
    # (hashable, stable) regardless of how the policy was constructed.
    p = EvidencePolicy(allowed_transforms=["json_rows"])
    assert p.allowed_transforms == ("json_rows",)
    assert isinstance(p.allowed_transforms, tuple)


def test_policy_has_max_output_tokens_defaulting_to_none():
    assert EvidencePolicy().max_output_tokens is None
