"""Guard test: EvidencePolicy (runtime) and EvidencePolicyConfig (config) must
expose the same field name set.

A field added to one type but not the other fails this test immediately, rather
than being caught only at serialization/deserialization time in production.
"""

from drawbore.config.models import EvidencePolicyConfig
from drawbore.evidence.policy import EvidencePolicy


def test_field_name_sets_are_identical() -> None:
    runtime_fields = set(EvidencePolicy.model_fields.keys())
    config_fields = set(EvidencePolicyConfig.model_fields.keys())

    only_in_runtime = runtime_fields - config_fields
    only_in_config = config_fields - runtime_fields

    assert only_in_runtime == set(), (
        f"Fields present in EvidencePolicy but missing from EvidencePolicyConfig: "
        f"{sorted(only_in_runtime)}"
    )
    assert only_in_config == set(), (
        f"Fields present in EvidencePolicyConfig but missing from EvidencePolicy: "
        f"{sorted(only_in_config)}"
    )
