"""Deterministic evidence transforms.

A transform is a named, deterministic strategy: ``should_apply`` decides whether
it fits the value + policy; ``compress`` returns ``(compressed_value, warnings)``
where the same input + policy yields byte-identical output. Transforms operate on a
plain ``dict``/``list`` (the agent input's ``model_dump()``); the pipeline
re-validates the compressed view against the input model, so a transform must keep
the structure schema-compatible.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from .policy import EvidencePolicy
from .tokens import estimate_tokens

# json_rows tuning (module constants so behaviour is explicit + deterministic).
_HEAD = 5
_TAIL = 5
_SAMPLE = 10
# A row is "notable" when a structured signal field holds a non-benign value.
_SIGNAL_FIELDS = ("status", "error", "flagged", "high_risk", "anomaly", "risk")
_BENIGN_VALUES = ("ok", "pass", "passed", "none", "false", "low", "clear", "")


@runtime_checkable
class EvidenceTransform(Protocol):
    name: str
    content_type: str

    def should_apply(self, value: Any, policy: EvidencePolicy) -> bool: ...
    def compress(self, value: Any, policy: EvidencePolicy) -> tuple[Any, tuple[str, ...]]: ...


def _is_notable(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    for field in _SIGNAL_FIELDS:
        if field in row:
            value = row[field]
            if isinstance(value, bool):
                if value:
                    return True
            elif isinstance(value, str):
                if value.strip().lower() not in _BENIGN_VALUES:
                    return True
            elif value not in (None, 0):
                return True
    return False


def _row_budget(policy: EvidencePolicy, *, n_lists: int) -> int | None:
    """The per-list row cap derived from the output token budget, divided evenly
    across the dict's compressible lists (so the whole view honours the budget).
    ``None`` when no budget is set. Floor of 1 so a tiny budget still keeps a row.
    Deterministic; ~40 tokens/row."""
    if policy.max_output_tokens is None:
        return None
    total_rows = max(1, policy.max_output_tokens // 40)
    return max(1, total_rows // max(1, n_lists))


def _select_items(
    items: list[Any],
    max_keep: int | None,
    *,
    head: int,
    tail: int,
    is_signal: Callable[[Any], bool],
    sample: int,
    cap_msg: str,
    signal_note: str,
    drop_msg: str,
) -> tuple[list[Any], tuple[str, ...]]:
    """Keep head + tail + signal + optional evenly-spaced sample, in original order,
    deduped by index, optionally capped. Deterministic. ``sample=0`` disables sampling.
    Message templates use ``{dropped}``, ``{n}``, and ``{signal_dropped}`` as named slots."""
    n = len(items)
    keep_idx: set[int] = set(range(min(head, n)))
    keep_idx |= set(range(max(0, n - tail), n))
    keep_idx |= {i for i, x in enumerate(items) if is_signal(x)}
    if sample > 0 and n > 0:
        stride = max(1, n // sample)
        keep_idx |= set(range(0, n, stride))
    ordered = sorted(keep_idx)
    warnings: list[str] = []
    if max_keep is not None and len(ordered) > max_keep:
        # Deterministic cap: keep the first max_keep selected indices (head-biased).
        dropped_idx = ordered[max_keep:]
        signal_dropped = sum(1 for i in dropped_idx if is_signal(items[i]))
        ordered = ordered[:max_keep]
        msg = cap_msg.format(dropped=len(dropped_idx))
        if signal_dropped:
            msg += signal_note.format(signal_dropped=signal_dropped)
        warnings.append(msg)
    if len(ordered) < n:
        warnings.append(drop_msg.format(dropped=n - len(ordered), n=n))
    return [items[i] for i in ordered], tuple(warnings)


def _compress_rows(rows: list[Any], max_keep: int | None) -> tuple[list[Any], tuple[str, ...]]:
    """Keep head + tail + notable rows + an evenly-spaced sample, in original order,
    deduped by index, optionally capped. Deterministic."""
    return _select_items(
        rows, max_keep,
        head=_HEAD, tail=_TAIL, is_signal=_is_notable, sample=_SAMPLE,
        cap_msg="capped evidence rows: dropped {dropped} selected rows over the output cap",
        # Legibility: an auditor must see that *signal* was dropped,
        # not just a row count. The originals stay retrievable through the proxy.
        signal_note=" (including {signal_dropped} notable/signal rows)",
        drop_msg="dropped {dropped} of {n} rows (originals retained for retrieval)",
    )


class _JsonRows:
    name = "json_rows"
    content_type = "json_rows"

    def should_apply(self, value: Any, policy: EvidencePolicy) -> bool:
        if not _has_compressible_list(value):
            return False
        return estimate_tokens(value) >= policy.min_tokens

    def compress(self, value: Any, policy: EvidencePolicy) -> tuple[Any, tuple[str, ...]]:
        # max rows derived from the output token budget (deterministic), if any.
        if isinstance(value, list):
            return _compress_rows(value, _row_budget(policy, n_lists=1))
        if isinstance(value, dict):
            # The budget is divided across the dict's compressible lists so the
            # whole compressed view honours max_output_tokens, not each list alone.
            keys = [k for k, v in value.items() if isinstance(v, list) and _is_record_list(v)]
            max_keep = _row_budget(policy, n_lists=len(keys))
            out = dict(value)
            warnings: list[str] = []
            for key in keys:
                kept, w = _compress_rows(value[key], max_keep)
                out[key] = kept
                warnings.extend(w)
            return out, tuple(warnings)
        return value, ()


json_rows: EvidenceTransform = _JsonRows()

_REGISTRY: dict[str, EvidenceTransform] = {json_rows.name: json_rows}


def get_transform(name: str) -> EvidenceTransform | None:
    """Return the registered transform for ``name`` (or None)."""
    return _REGISTRY.get(name)


def register_transform(transform: EvidenceTransform) -> None:
    """Register a transform under its ``name`` (used by `logs` in a later task)."""
    _REGISTRY[transform.name] = transform


def _is_record_list(value: list[Any]) -> bool:
    return len(value) > 0 and all(isinstance(item, dict) for item in value)


def _has_compressible_list(value: Any) -> bool:
    if isinstance(value, list):
        return _is_record_list(value)
    if isinstance(value, dict):
        return any(isinstance(v, list) and _is_record_list(v) for v in value.values())
    return False


# logs tuning.
_LOG_HEAD = 10
_LOG_TAIL = 10
_SEVERITY_MARKERS = ("ERROR", "WARN", "CRITICAL", "FATAL", "EXCEPTION", "TRACEBACK")
# Only the traceback *frame* line, which is uniquely shaped ("  File ...").
# We deliberately do NOT treat bare indentation ("    "/"\t") as severe: doing so
# would mark every indented line of a pretty-printed/JSON/YAML log as severe and
# defeat compression entirely. The "Traceback ..." header is caught by the
# TRACEBACK severity marker; the exception line, if present, is tail-preserved.
_TRACE_PREFIXES = ("  File ",)


def _is_severe_line(line: str) -> bool:
    stripped = line.lstrip()
    upper = stripped.upper()
    # Line-level severity: a marker at the START of the (stripped) line, or a
    # traceback frame line — NOT a benign mid-sentence mention of "error" and NOT
    # mere indentation.
    if any(upper.startswith(marker) for marker in _SEVERITY_MARKERS):
        return True
    return any(line.startswith(prefix) for prefix in _TRACE_PREFIXES)


def _compress_log_lines(lines: list[str], max_keep: int | None) -> tuple[list[str], tuple[str, ...]]:
    return _select_items(
        lines, max_keep,
        head=_LOG_HEAD, tail=_LOG_TAIL, is_signal=_is_severe_line, sample=0,
        cap_msg="capped evidence log lines: dropped {dropped} over the output cap",
        signal_note=" (including {signal_dropped} severity/error lines)",
        drop_msg="dropped {dropped} of {n} log lines (original retained for retrieval)",
    )


class _Logs:
    name = "logs"
    content_type = "logs"

    def _log_fields(self, value: Any) -> list[str]:
        # A log field is a newline-delimited string value: that is what this
        # transform compresses (line by line). A single newline-free blob is
        # intentionally out of scope — it has no deterministic line structure to
        # compress, so it passes through untouched (json_rows handles record lists).
        if not isinstance(value, dict):
            return []
        return [k for k, v in value.items() if isinstance(v, str) and "\n" in v]

    def should_apply(self, value: Any, policy: EvidencePolicy) -> bool:
        if not self._log_fields(value):
            return False
        return estimate_tokens(value) >= policy.min_tokens

    def compress(self, value: Any, policy: EvidencePolicy) -> tuple[Any, tuple[str, ...]]:
        fields = self._log_fields(value)
        max_keep = None
        if policy.max_output_tokens is not None:
            # ~20 tokens/line, divided across the dict's log fields (honour the budget).
            total_lines = max(1, policy.max_output_tokens // 20)
            max_keep = max(1, total_lines // len(fields))
        out = dict(value)
        warnings: list[str] = []
        for key in fields:
            lines = value[key].split("\n")
            kept, w = _compress_log_lines(lines, max_keep)
            out[key] = "\n".join(kept)
            warnings.extend(w)
        return out, tuple(warnings)


logs: EvidenceTransform = _Logs()
register_transform(logs)
