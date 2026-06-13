"""Golden byte-identity tests: the config fingerprints must never change bytes
when their hashing is refactored, and the shared canonical-hash leaf must
reproduce the exact same algorithm.

Triage note for a failing golden
---------------------------------
A failing golden fingerprint test has two possible causes:

(a) A Drawbore hashing refactor changed the algorithm — the
    ``test_pydantic_version_matches_golden_baseline`` test PASSES.  This is a real
    regression: do NOT update the golden; investigate the change in
    ``src/drawbore/_canon.py`` or ``src/drawbore/config/fingerprint.py``.

(b) A Pydantic version change shifted ``model_json_schema()`` output — the
    ``test_pydantic_version_matches_golden_baseline`` test ALSO FAILS.  This is not
    a Drawbore regression; it means the JSON schema serialization changed across
    Pydantic versions.  Before re-pinning the goldens, investigate the cross-version
    impact: any persisted ``StepSeal``, ``PipelineConfig`` manifest, or re-attestation
    surface fingerprint computed under the old version will no longer compare equal
    under the new version, producing false-positive ``resume_drift`` halts, phantom
    authority regressions, or spurious forced re-attestations.  See
    ``docs/guide/reliability.mdx`` for the pinning guidance.
"""

import importlib.metadata
from typing import Literal, Optional

from pydantic import BaseModel, Field

from drawbore._canon import canonical_fingerprint, text_fingerprint
from drawbore.config.fingerprint import schema_fingerprint, footprint_fingerprint
from drawbore.identity.identity import _schema_fingerprint as identity_schema_fingerprint

# ---------------------------------------------------------------------------
# Pydantic version baseline
# ---------------------------------------------------------------------------

# The Pydantic version these golden fingerprints were recorded against.
# If a golden_*_bytes test below fails, FIRST check whether this differs from the
# installed version: a mismatch means Pydantic changed model_json_schema() output
# (a version drift to triage, see the module docstring), NOT a Drawbore refactor.
GOLDEN_PYDANTIC_VERSION = "2.13.4"


def test_pydantic_version_matches_golden_baseline():
    installed = importlib.metadata.version("pydantic")
    assert installed == GOLDEN_PYDANTIC_VERSION, (
        f"Pydantic {installed} != golden baseline {GOLDEN_PYDANTIC_VERSION}; "
        "if a golden fingerprint test also fails, it is a Pydantic schema-output "
        "shift, not a Drawbore regression — see this module's triage note."
    )


# ---------------------------------------------------------------------------
# Models used by the golden tests
# ---------------------------------------------------------------------------


class _Order(BaseModel):
    order_id: str
    amount: float


class _WithLiteral(BaseModel):
    decision: Literal["confirmed", "blocked", "needs_review"]


class _WithTuple(BaseModel):
    coords: tuple[float, float, float]


class _WithListDict(BaseModel):
    rows: list[dict]


class _Inner(BaseModel):
    value: int


class _Nested(BaseModel):
    inner: _Inner
    label: str


class _WithOptional(BaseModel):
    note: Optional[str] = None
    count: int | None = None


class _WithConstrained(BaseModel):
    name: str = Field(max_length=100)
    score: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Existing goldens (original)
# ---------------------------------------------------------------------------


def test_schema_fingerprint_golden_bytes():
    # Pinned literal — recorded from the original implementation.
    # If this fails after a refactor, the canonical form changed — that is a
    # regression, not a test to update.
    assert schema_fingerprint(_Order) == (
        "sha256:79511b14725f07e04768ed4cba63aaa7566577ef22c293bb4aba4689039cf1a4"
    )


def test_footprint_fingerprint_golden_bytes():
    # Pinned literal — recorded from the original implementation.
    facts = [("a", "tool", "t1", "read"), ("b", "tool", "t2", "write")]
    assert footprint_fingerprint(facts) == (
        "sha256:80eba630acd86befa5197fd1c2233885970373d91f4dc19f5e738bef1fb3580e"
    )


def test_canonical_fingerprint_matches_schema_fingerprint_algorithm():
    assert canonical_fingerprint(_Order.model_json_schema()) == schema_fingerprint(_Order)


def test_canonical_fingerprint_key_order_insensitive():
    assert canonical_fingerprint({"b": 1, "a": 2}) == canonical_fingerprint({"a": 2, "b": 1})


def test_text_fingerprint_none_is_distinct_sentinel():
    # None must NOT hash like the empty string: a None -> "" change is drift.
    assert text_fingerprint(None) == "none"
    assert text_fingerprint("") != "none"
    assert text_fingerprint("") != text_fingerprint(None)
    assert text_fingerprint("abc").startswith("sha256:")


def test_canon_is_stdlib_only():
    import drawbore._canon as canon

    source = open(canon.__file__).read()
    assert "pydantic" not in source
    assert "drawbore." not in source.replace("drawbore._canon", "")


# ---------------------------------------------------------------------------
# Richer goldens: Pydantic-sensitive schema constructs
# ---------------------------------------------------------------------------


def test_schema_fingerprint_golden_literal():
    # Literal fields: encoding can change across Pydantic minors
    # (enum vs const representation, "type" annotation presence).
    assert schema_fingerprint(_WithLiteral) == (
        "sha256:7917b42e2cb33fa4cf936a05c0fd11acfaaed121fd0e1fc8b52b1662837bd395"
    )


def test_schema_fingerprint_golden_tuple():
    # Fixed-length tuple/sequence fields: "prefixItems" vs "items" encoding.
    assert schema_fingerprint(_WithTuple) == (
        "sha256:fd2cefae0ccd422f8e2cabe9567a32344f0ed46880309ea9a534bf62d41e9488"
    )


def test_schema_fingerprint_golden_list_dict():
    # list[dict]: bare dict encoding may vary (additionalProperties presence).
    assert schema_fingerprint(_WithListDict) == (
        "sha256:adf39cff0c95ae79606a6099e9d252a88bf7dcd676a660c7d76ae7660a61cd48"
    )


def test_schema_fingerprint_golden_nested():
    # Nested BaseModel: exercises $defs and $ref resolution; key ordering
    # in $defs can shift across Pydantic minors.
    assert schema_fingerprint(_Nested) == (
        "sha256:6c41d23840a9710516f460787ff0178f4986a27251fe843b347043aa5400a8fe"
    )


def test_schema_fingerprint_golden_optional_union():
    # Optional / X | None union: anyOf encoding and null representation.
    assert schema_fingerprint(_WithOptional) == (
        "sha256:d22677c40a392d705498e9b54abece839c8c9ba5c72cf4e89c88312c924ba967"
    )


def test_schema_fingerprint_golden_constrained():
    # Constrained fields (max_length, ge/le): keyword placement may vary.
    assert schema_fingerprint(_WithConstrained) == (
        "sha256:086753522f1b6bcfcc7ed180cac76a1fe12f5d7946c9b38b05fa2f3e33ce7c62"
    )


# ---------------------------------------------------------------------------
# Identity re-attestation fingerprint path
# ---------------------------------------------------------------------------


def test_identity_schema_fingerprint_golden():
    # identity uses raw sorted JSON (not sha256) for the re-attestation surface;
    # a Pydantic schema-output shift here forces spurious re-attestation even
    # without any schema change by the developer.
    assert identity_schema_fingerprint(_WithLiteral) == (
        '{"properties": {"decision": {"enum": ["confirmed", "blocked", "needs_review"],'
        ' "title": "Decision", "type": "string"}}, "required": ["decision"],'
        ' "title": "_WithLiteral", "type": "object"}'
    )
