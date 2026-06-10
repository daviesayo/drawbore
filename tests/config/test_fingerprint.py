import json

from pydantic import BaseModel

from drawbore.config.fingerprint import schema_fingerprint
from drawbore.observability import payload_hash


class _Model(BaseModel):
    amount: int
    currency: str


def test_fingerprint_is_canonical_full_sha256_with_prefix():
    fp = schema_fingerprint(_Model)
    assert fp.startswith("sha256:")
    hexpart = fp.split(":", 1)[1]
    assert len(hexpart) == 64                      # full SHA-256, not truncated
    int(hexpart, 16)                               # is hex


def test_fingerprint_matches_an_independent_sha256_oracle():
    # Pinned against an EXTERNAL oracle (the system `shasum -a 256`), not the
    # implementation's own hashlib call, so a regression in the canonical form
    # (dropping sort_keys, wrong separators, wrong encoding) is actually caught
    # rather than being mirrored by a re-derivation. Provenance:
    #   python -c '...print(json.dumps(_Model.model_json_schema(),
    #              sort_keys=True, separators=(",",":")), end="")' | shasum -a 256
    # over canonical: {"properties":{"amount":{"title":"Amount","type":"integer"},
    #   "currency":{"title":"Currency","type":"string"}},"required":["amount",
    #   "currency"],"title":"_Model","type":"object"}
    assert schema_fingerprint(_Model) == (
        "sha256:03aa36ed2e08d92109982d345cf021cccf0f8ded0bb7731d254f36b82b060f4d"
    )


def test_fingerprint_is_not_payload_hash():
    # The schema-drift fingerprint must NOT be the truncated observability
    # payload_hash (a short span/audit correlation id), neither in value nor length.
    canonical = json.dumps(_Model.model_json_schema(), sort_keys=True, separators=(",", ":"))
    assert schema_fingerprint(_Model) != payload_hash(_Model.model_json_schema())
    assert schema_fingerprint(_Model).split(":", 1)[1] != payload_hash(canonical)


def test_fingerprint_changes_when_the_schema_changes():
    class _Other(BaseModel):
        amount: int
        currency: str
        memo: str

    assert schema_fingerprint(_Model) != schema_fingerprint(_Other)


from drawbore.config.fingerprint import footprint_fingerprint  # noqa: E402


def test_footprint_fingerprint_matches_independent_oracle():
    # Pinned: printf '%s' '[["a","tool","ref1","*"],["b","tool","ref2","*"]]' | shasum -a 256
    # Canonical form: json.dumps(sorted([list(t) for t in facts]), separators=(",",":"))
    # → [["a","tool","ref1","*"],["b","tool","ref2","*"]]
    facts = [("b", "tool", "ref2", "*"), ("a", "tool", "ref1", "*")]  # unsorted input on purpose
    assert footprint_fingerprint(facts) == (
        "sha256:0086bccfab6c6714eb6f106d9800806a36a62ef74fdcf98260a78fea6e357323"
    )
