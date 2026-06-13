"""Payload identity hashing.

One scheme shared by OTel span tags, the tool-proxy log, and audit records, so an
input/output hash is comparable across a span, a proxy-log entry, and the audit
trail. ``sha256(repr(value))`` truncated to 16 hex chars — stable and short, not a
security primitive (it identifies a payload, it does not protect it)."""

from __future__ import annotations

import hashlib
from typing import Any


# Three hashers serve distinct purposes — do not mix them up:
# ``payload_hash`` (here): short repr-based identity tag for spans/proxy-log/audit; NOT order-stable, NOT for fingerprinting.
# ``canonical_fingerprint``/``text_fingerprint`` (drawbore._canon): config-drift and step-seal fingerprints; NOT for short display tags.
# ``ledger_args_hash`` (drawbore.state.effect_ledger): order-stable args hash for the effect ledger; NOT for payload identity or config drift.
def payload_hash(value: Any) -> str:
    """Return a short, stable identity for ``value`` (first 16 hex of its
    sha256(repr))."""
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:16]
