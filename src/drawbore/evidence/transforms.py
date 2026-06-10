"""Deterministic evidence transforms.

A transform is a named, deterministic strategy: ``should_apply`` decides whether
it fits the value + policy; ``compress`` returns ``(compressed_value, warnings)``
where the same input + policy yields byte-identical output. Transforms operate on a
plain ``dict``/``list`` (the agent input's ``model_dump()``); the pipeline
re-validates the compressed view against the input model, so a transform must keep
the structure schema-compatible.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

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


def _compress_rows(rows: list[Any], max_keep: int | None) -> tuple[list[Any], tuple[str, ...]]:
    """Keep head + tail + notable rows + an evenly-spaced sample, in original order,
    deduped by index, optionally capped. Deterministic."""
    n = len(rows)
    keep_idx: set[int] = set(range(min(_HEAD, n)))
    keep_idx |= set(range(max(0, n - _TAIL), n))
    keep_idx |= {i for i, row in enumerate(rows) if _is_notable(row)}
    if _SAMPLE > 0 and n > 0:
        stride = max(1, n // _SAMPLE)
        keep_idx |= set(range(0, n, stride))
    ordered = sorted(keep_idx)
    warnings: list[str] = []
    if max_keep is not None and len(ordered) > max_keep:
        # Deterministic cap: keep the first max_keep selected indices (head-biased).
        dropped_idx = ordered[max_keep:]
        notable_dropped = sum(1 for i in dropped_idx if _is_notable(rows[i]))
        ordered = ordered[:max_keep]
        msg = f"capped evidence rows: dropped {len(dropped_idx)} selected rows over the output cap"
        if notable_dropped:
            # Legibility: an auditor must see that *signal* was dropped,
            # not just a row count. The originals stay retrievable through the proxy.
            msg += f" (including {notable_dropped} notable/signal rows)"
        warnings.append(msg)
    if len(ordered) < n:
        warnings.append(f"dropped {n - len(ordered)} of {n} rows (originals retained for retrieval)")
    return [rows[i] for i in ordered], tuple(warnings)


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
    n = len(lines)
    keep_idx: set[int] = set(range(min(_LOG_HEAD, n)))
    keep_idx |= set(range(max(0, n - _LOG_TAIL), n))
    keep_idx |= {i for i, line in enumerate(lines) if _is_severe_line(line)}
    ordered = sorted(keep_idx)
    warnings: list[str] = []
    if max_keep is not None and len(ordered) > max_keep:
        dropped_idx = ordered[max_keep:]
        severe_dropped = sum(1 for i in dropped_idx if _is_severe_line(lines[i]))
        ordered = ordered[:max_keep]
        msg = f"capped evidence log lines: dropped {len(dropped_idx)} over the output cap"
        if severe_dropped:
            msg += f" (including {severe_dropped} severity/error lines)"
        warnings.append(msg)
    if len(ordered) < n:
        warnings.append(f"dropped {n - len(ordered)} of {n} log lines (original retained for retrieval)")
    return [lines[i] for i in ordered], tuple(warnings)


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
        if policy.max_output_tokens is not None and fields:
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
