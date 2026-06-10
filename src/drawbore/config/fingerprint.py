"""Schema drift fingerprint.

A compact, deterministic, FULL SHA-256 over a model's canonical JSON schema. This
follows the deterministic schema-serialization precedent in
``identity._schema_fingerprint`` but upgrades it to a full hash for drift checks.

NOT ``observability.payload_hash`` — that is a short payload-identity for span/audit
correlation, not a schema-drift oracle. And NOT the type-assignability oracle
either: these hashes are review/drift evidence; ``Pipeline.add`` still owns
static compatibility.
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

from pydantic import BaseModel

from drawbore._canon import canonical_fingerprint


def schema_fingerprint(model: type[BaseModel]) -> str:
    """``"sha256:" + sha256(canonical-json(model.model_json_schema()))``.

    Canonical = ``sort_keys=True`` + compact separators, so the fingerprint is
    stable across runs and Python dict orderings.
    """
    return canonical_fingerprint(model.model_json_schema())


def footprint_fingerprint(facts: Iterable[tuple[str, str, str, str]]) -> str:
    """``"sha256:" + sha256(canonical-json(sorted facts))``.

    Takes raw ``(subject, kind, ref, scope)`` tuples (not the CapabilityFootprint
    type) so this module keeps no dependency on ``authority`` — same canonical-JSON
    + full-SHA-256 scheme as ``schema_fingerprint``.
    """
    rows = sorted([list(t) for t in facts])
    canonical = json.dumps(rows, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
