"""Canonical-JSON SHA-256 hashing shared across subsystems.

One algorithm, one implementation: ``sort_keys=True`` + compact separators,
prefixed ``"sha256:"``. Used by the config drift fingerprints and the resume
step seal so the two never diverge. Stdlib only — this module must not import
Pydantic or any drawbore subsystem.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_fingerprint(obj: Any) -> str:
    """``"sha256:" + sha256(canonical-json(obj))``.

    Canonical = ``sort_keys=True`` + compact separators, so the fingerprint is
    stable across runs and Python dict orderings.
    """
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def text_fingerprint(text: str | None) -> str:
    """Fingerprint for an optional text field.

    ``None`` maps to the literal sentinel ``"none"`` — distinct from the hash
    of the empty string, so a ``None -> ""`` change is still detectable.
    """
    if text is None:
        return "none"
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
