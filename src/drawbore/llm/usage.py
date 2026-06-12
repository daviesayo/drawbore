"""Token usage and cost read off a model response.

Owned by ``drawbore.llm`` — the model boundary. ``TokenUsage`` is the framework's
provider-agnostic token-count value (``input``/``output``/``total``, never a
provider's own field names). ``extract_usage`` / ``extract_cost`` read these off a
gateway response defensively: a response that does not report tokens or cost yields
``None`` (never a fabricated number). Cost is taken only from what the response
already carries — it is never computed from a pricing table here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    """Provider-agnostic token counts for one model call. ``input_tokens`` is the
    prompt side, ``output_tokens`` the generated side, ``total_tokens`` their sum (or
    the provider-reported total when supplied)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


def _field(obj: Any, key: str) -> Any:
    """Read ``key`` off a dict-or-object usage container, returning None if absent."""
    if isinstance(obj, dict):
        return obj.get(key)
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            pass
    return getattr(obj, key, None)


def extract_usage(response: Any) -> TokenUsage | None:
    """Read token usage off a gateway/provider response. The response reports usage
    as ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` (object or dict
    access). Returns ``None`` when the response reports no usage at all — never a
    zero-filled placeholder."""
    usage = _field(response, "usage")
    if usage is None:
        return None
    prompt = _field(usage, "prompt_tokens")
    completion = _field(usage, "completion_tokens")
    total = _field(usage, "total_tokens")
    if prompt is None and completion is None and total is None:
        return None
    try:
        input_tokens = int(prompt) if prompt is not None else 0
        output_tokens = int(completion) if completion is not None else 0
        total_tokens = int(total) if total is not None else input_tokens + output_tokens
    except (TypeError, ValueError):
        return None
    return TokenUsage(
        input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens
    )


def extract_cost(response: Any) -> float | None:
    """Read the provider-reported cost the response already carries (on its hidden
    metadata as ``response_cost``). Returns ``None`` when no cost is present — cost is
    never computed from a pricing table here, so an unpriced call stays ``None``."""
    hidden = getattr(response, "_hidden_params", None)
    if not isinstance(hidden, dict):
        return None
    cost = hidden.get("response_cost")
    if cost is None:
        return None
    try:
        return float(cost)
    except (TypeError, ValueError):
        return None
