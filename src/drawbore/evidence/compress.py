"""The pre-model compression entrypoint.

``compress_for_model`` is a pure function: given the payload (the agent input's
``model_dump()``) + a policy + a store, it routes to a transform, honours the mode
(simulate/compress), gates on ``min_tokens``, stores the original (failing closed
if it cannot), and returns ``(model_view, decision, handle)``. It imports no
pipeline/audit/observability/schema — the pipeline re-validates ``model_view``
against the input model and records the decision (layering).
"""

from __future__ import annotations

import hashlib
from typing import Any

from .errors import EvidenceStoreError, EvidenceTransformError
from .policy import EvidencePolicy
from .records import EvidenceDecision, EvidenceHandle
from .store import EvidenceStore
from .tokens import _canonical_json, estimate_tokens
from .transforms import get_transform


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()[:16]


def _passthrough(policy: EvidencePolicy, payload: Any, reason: str) -> tuple[Any, EvidenceDecision, None]:
    return payload, EvidenceDecision(
        decision="passthrough", reason=reason, policy=policy.name, transform=None,
        original_tokens=estimate_tokens(payload), compressed_tokens=None, handle_id=None,
    ), None


def _select_transform(payload: Any, policy: EvidencePolicy):
    for name in policy.allowed_transforms:
        transform = get_transform(name)
        if transform is not None and transform.should_apply(payload, policy):
            return transform
    return None


def compress_for_model(
    payload: Any,
    policy: EvidencePolicy,
    *,
    store: EvidenceStore,
    run_id: str,
    step: int,
    source_agent: str,
) -> tuple[Any, EvidenceDecision, EvidenceHandle | None]:
    """Return ``(model_view, decision, handle)``. ``model_view`` is what the model
    should see; on passthrough it is ``payload`` unchanged. Fails closed
    (``EvidenceStoreError``) if compression is enabled, the original is required,
    and storage fails."""
    if not policy.enabled:
        return _passthrough(policy, payload, "policy disabled")

    original_tokens = estimate_tokens(payload)
    if original_tokens < policy.min_tokens:
        return _passthrough(policy, payload, f"below min_tokens ({original_tokens} < {policy.min_tokens})")

    transform = _select_transform(payload, policy)
    if transform is None:
        return _passthrough(policy, payload, "no applicable transform")

    # Hash the original ONCE, before the transform runs — so handle_id and
    # original_hash agree and stay correct even if a future transform mutates input.
    original_hash = _hash(payload)
    try:
        compressed, warnings = transform.compress(payload, policy)
    except EvidenceTransformError:
        raise  # an intentional transform failure escalates as-is (legible)
    except Exception as exc:  # an unexpected error in the transform is a transform failure
        raise EvidenceTransformError(f"transform '{transform.name}' failed: {exc}") from exc

    compressed_tokens = estimate_tokens(compressed)

    if policy.mode == "simulate":
        # Record what WOULD happen; do not build a handle, change the payload, or store.
        return payload, EvidenceDecision(
            decision="compressed", reason="simulate (payload unchanged)", policy=policy.name,
            transform=transform.name, original_tokens=original_tokens,
            compressed_tokens=compressed_tokens, handle_id=None, warnings=warnings,
        ), None

    handle = EvidenceHandle(
        handle_id=_handle_id(run_id, step, source_agent, transform.name, original_hash),
        run_id=run_id, step=step, source_agent=source_agent,
        content_type=transform.content_type, original_hash=original_hash,
        compressed_hash=_hash(compressed), original_tokens=original_tokens,
        compressed_tokens=compressed_tokens, transform=transform.name,
    )

    # mode == "compress": store the original (fail closed) then return the view.
    try:
        store.put(handle, original=payload, compressed=compressed, ttl_seconds=policy.ttl_seconds)
    except Exception as exc:
        if policy.require_original_store:
            raise EvidenceStoreError(
                f"cannot store original evidence (compression aborted, originals are "
                f"always retained): {exc}"
            ) from exc
        return _passthrough(policy, payload, f"store unavailable; passthrough: {exc}")

    return compressed, EvidenceDecision(
        decision="compressed", reason="compressed for model", policy=policy.name,
        transform=transform.name, original_tokens=original_tokens,
        compressed_tokens=compressed_tokens, handle_id=handle.handle_id, warnings=warnings,
    ), handle


def _handle_id(run_id: str, step: int, source_agent: str, transform: str, original_hash: str) -> str:
    seed = f"{run_id}:{step}:{source_agent}:{transform}:{original_hash}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]
