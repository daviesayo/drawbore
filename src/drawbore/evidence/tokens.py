"""Deterministic token estimate.

NOT a real tokenizer — Drawbore adds no tokenizer dependency. ``estimate_tokens``
is a stable size proxy (``len(json.dumps(value)) // 4`` — the common ~4-chars-per-
token heuristic), so policy thresholds (``min_tokens``) and handle metadata are
deterministic and dependency-free. ``default=str`` keeps it total over non-JSON-
native values (datetimes/Decimals carried by ``model_dump()``)."""

from __future__ import annotations

import json
from typing import Any


def estimate_tokens(value: Any) -> int:
    """Return a deterministic, dependency-free token estimate for ``value`` (at
    least 1)."""
    serialized = json.dumps(value, sort_keys=True, default=str)
    return max(1, len(serialized) // 4)
