"""Evidence compression policy.

Declared per-step on ``Pipeline.add(agent, ..., evidence=EvidencePolicy(...))``.
``enabled=False`` (the default) means byte-identical passthrough — the feature is
opt-in. ``mode`` governs behavior WHEN enabled: ``"simulate"`` records
the decision without changing the payload; ``"compress"`` actually compresses.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class EvidencePolicy(BaseModel):
    """When and how a step's model-bound evidence may be compressed."""

    name: str = "default"
    enabled: bool = False
    mode: Literal["simulate", "compress"] = "compress"
    allowed_transforms: tuple[str, ...] = ("json_rows", "logs")
    min_tokens: int = 800
    # Output token budget a transform may compress toward (None = no cap).
    max_output_tokens: int | None = None
    require_original_store: bool = True
    allow_full_retrieval: bool = False
    allow_search_retrieval: bool = True
    ttl_seconds: int | None = None
    strict: bool = False
