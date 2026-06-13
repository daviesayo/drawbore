"""Directional input-schema relaxation oracle. Pure config layer.

A static, fail-closed comparator over two ``PipelineConfig`` manifests. For every
agent present in BOTH manifests (matched by name) it compares the ``input_schema``
JSON-schema dicts and flags any change that *relaxes* the input gate — i.e. makes
the agent accept a strictly wider set of inputs than before. A wider accepted-input
set is a safety regression on the data axis: the validation gate just got weaker.

Soundness contract: the comparator must NEVER pass a genuine relaxation. It MAY
flag a safe change it cannot *prove* is a narrowing (the fail-closed direction —
annoying, never unsafe). The governing rule is "prove narrowing-or-equal, else
flag."

Imports only the config models and the shared canonical fingerprint; no
tools/runtime/engine imports (the config import boundary).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from drawbore._canon import canonical_fingerprint

from .errors import SchemaRelaxationError
from .models import PipelineConfig

# The exhaustive metadata ignore-list. A keyword present ONLY in old (removed) is a
# relaxation UNLESS it is one of these — these annotate, they do not constrain the
# accepted-input set. This is the ONLY pass-list for removals; every other removed
# keyword is flagged (fail-closed).
_METADATA = frozenset({"title", "description", "default", "examples"})

# Length / count bounds where an INCREASE widens the accepted set.
_UPPER_BOUNDS = frozenset({"maxLength", "maxItems", "maxProperties", "maximum", "exclusiveMaximum"})
# Length / count / numeric bounds where a DECREASE widens the accepted set.
_LOWER_BOUNDS = frozenset({"minLength", "minItems", "minProperties", "minimum", "exclusiveMinimum"})

# Keywords whose containment / equivalence is undecidable or whose any-change
# direction we will not try to prove: ANY change (value present in both, differing)
# fails closed.
_OPAQUE = frozenset(
    {"pattern", "type", "format", "$ref", "anyOf", "oneOf", "allOf", "not", "if", "then", "else"}
)

# Sub-schema container keywords: when present in both with a differing value they
# are descended structurally rather than compared as opaque scalars.
_CONTAINERS = frozenset({"properties", "$defs", "definitions", "items", "additionalProperties"})


@dataclass(frozen=True)
class SchemaRelaxation:
    """One flagged relaxation at a concrete JSON-schema path."""

    agent: str   # name-matched agent whose input schema relaxed
    path: str    # actual dict-traversal route, e.g. "/$defs/Address/properties/zip/maxLength"
    keyword: str # the JSON-schema keyword involved ("maxLength" | "required" | "enum" | ...)
    old: Any     # old value (None if absent)
    new: Any     # new value (None if absent)
    reason: str  # auditor-readable one-liner


@dataclass(frozen=True)
class SchemaRelaxationDiff:
    old_fingerprint: str  # canonical_fingerprint({a.name: a.input_schema for a in old.agents})
    new_fingerprint: str  # same over new — covers exactly the input schemas this check reads
    relaxations: tuple[SchemaRelaxation, ...]  # empty => safe

    @property
    def ok(self) -> bool:
        return not self.relaxations

    def legible(self) -> str:
        status = "PASSED" if self.ok else "FAILED"
        lines = [
            f"Input-schema relaxation check: {status} "
            f"— {len(self.relaxations)} relaxation(s) detected."
        ]
        for r in self.relaxations:
            lines.append(
                f"- agent '{r.agent}' at {r.path}: '{r.keyword}' relaxed "
                f"(old={r.old!r} -> new={r.new!r}); {r.reason}"
            )
        lines.append(
            f"old_schemas {self.old_fingerprint}  new_schemas {self.new_fingerprint}"
        )
        return "\n".join(lines)

    def certificate(self) -> str:
        # Alias of legible() so a uniform verdict renderer can call certificate()
        # on either this diff or the authority diff.
        return self.legible()


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _num(x: Any) -> bool:
    # bool is a subclass of int; a boolean is never a numeric bound here.
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _divides(a: float, b: float) -> bool:
    """Does ``a`` divide ``b`` exactly (b is an integer multiple of a)?"""
    if a == 0:
        return False
    q = b / a
    return abs(q - round(q)) < 1e-9


def _value_set(values: Any) -> frozenset[str] | None:
    """Canonical-JSON set of an enum/const value-list, or None if not a list."""
    if not isinstance(values, list):
        return None
    return frozenset(json.dumps(v, sort_keys=True, separators=(",", ":")) for v in values)


def _ap_rank(value: Any) -> int:
    """Permissiveness of an ``additionalProperties`` value: False (none) < schema
    (constrained) < True (any). A higher rank on the new side is a loosening."""
    if value is False:
        return 0
    if value is True:
        return 2
    if isinstance(value, dict):
        return 1
    # Anything else (malformed) is treated as maximally permissive -> fail closed.
    return 2


def _join(path: str, seg: str) -> str:
    return f"{path}/{seg}"


# ---------------------------------------------------------------------------
# Recursive comparator (the heart)
# ---------------------------------------------------------------------------

class _SchemaComparator:
    """Carries the per-agent context (agent name and accumulator) so the recursive
    comparison methods only take ``path`` as a per-call argument."""

    def __init__(self, agent: str, out: list[SchemaRelaxation]) -> None:
        self._agent = agent
        self._out = out

    def _flag(self, path: str, keyword: str, ov: Any, nv: Any, reason: str) -> None:
        self._out.append(SchemaRelaxation(self._agent, path, keyword, ov, nv, reason))

    def compare(self, old: Any, new: Any, path: str) -> None:
        """Compare two schema nodes. Appends a SchemaRelaxation for every change that
        is provably or conservatively a relaxation (prove narrowing-or-equal, else flag)."""
        if not isinstance(old, dict) or not isinstance(new, dict):
            # A boolean schema or a category change at the node level. If it changed at
            # all and we cannot reason about it structurally, fail closed.
            if old != new:
                self._flag(
                    path or "/", "<node>", old, new,
                    "schema node changed to a non-object form; cannot prove narrowing",
                )
            return

        for key in sorted(set(old) | set(new)):
            in_old = key in old
            in_new = key in new

            if in_old and in_new:
                ov, nv = old[key], new[key]
                if ov == nv:
                    continue  # identical subtree: no relaxation possible
                if key in _METADATA:
                    continue  # annotations do not constrain the accepted-input set
                self._compare_changed(key, ov, nv, path, old)

            elif in_new:  # category 2: added keyword -> a new constraint can only narrow
                # (the field-presence exception is for added PROPERTY names, handled
                # inside _compare_properties; a new schema *keyword* always narrows).
                continue

            else:  # category 3: removed keyword -> flag unless on the metadata ignore-list
                if key in _METADATA:
                    continue
                self._flag(
                    _join(path, key), key, old[key], None,
                    "constraint removed; the accepted-input set may widen",
                )

    def _compare_changed(
        self,
        key: str,
        ov: Any,
        nv: Any,
        path: str,
        old_parent: dict[str, Any],
    ) -> None:
        """Category 1: a keyword present in BOTH whose value changed."""
        # --- sub-schema containers: descend structurally ---
        if key == "properties":
            self._compare_properties(ov, nv, path, old_parent.get("additionalProperties"))
            return
        if key in ("$defs", "definitions"):
            self._compare_defs(ov, nv, path, key)
            return
        if key == "items":
            self._compare_items(ov, nv, _join(path, "items"))
            return
        if key == "additionalProperties":
            self._compare_additional_properties(ov, nv, _join(path, key))
            return

        here = _join(path, key)

        def flag(reason: str) -> None:
            self._flag(here, key, ov, nv, reason)

        # --- decidable numeric / length / count bounds ---
        if key in _UPPER_BOUNDS:
            if _num(ov) and _num(nv):
                if nv > ov:
                    flag("upper bound raised; larger values now accepted")
                # nv < ov -> tightened -> pass
            else:
                flag("non-numeric upper-bound change; cannot prove narrowing")
            return
        if key in _LOWER_BOUNDS:
            if _num(ov) and _num(nv):
                if nv < ov:
                    flag("lower bound lowered; smaller values now accepted")
                # nv > ov -> tightened -> pass
            else:
                flag("non-numeric lower-bound change; cannot prove narrowing")
            return

        if key == "multipleOf":
            if _num(ov) and _num(nv) and ov > 0 and nv > 0:
                if _divides(nv, ov):       # new divides old -> multiples-of-new superset -> wider
                    flag("multipleOf widened; more values now divisible")
                elif _divides(ov, nv):     # old divides new -> narrower -> pass
                    pass
                else:
                    flag("multipleOf changed with no divisibility relation; some values newly accepted")
            else:
                flag("non-positive/non-numeric multipleOf change; cannot prove narrowing")
            return

        if key == "enum":
            old_set, new_set = _value_set(ov), _value_set(nv)
            if old_set is None or new_set is None:
                flag("malformed enum change; cannot prove narrowing")
            elif not new_set <= old_set:
                flag("enum admits value(s) absent from the old set")
            # new_set subset of old_set -> narrower -> pass
            return

        if key == "const":
            # A changed const means the new accepted value differs from the old one,
            # so a value old rejected is now accepted.
            flag("const value changed; a previously rejected value is now accepted")
            return

        if key == "required":
            if isinstance(ov, list) and isinstance(nv, list):
                removed = set(ov) - set(nv)
                if removed:
                    flag(f"required field(s) {sorted(removed)} dropped; now optional")
                # only additions -> narrower -> pass
            else:
                flag("malformed required change; cannot prove narrowing")
            return

        # --- fail-closed tail: opaque/undecidable keywords ---
        if key in _OPAQUE:
            flag("undecidable change; treated as a relaxation (fail-closed)")
            return

        # Any keyword we do not explicitly handle, whose value changed, fails closed.
        flag("unrecognised keyword changed; treated as a relaxation (fail-closed)")

    def _compare_properties(
        self,
        ov: Any,
        nv: Any,
        path: str,
        old_additional: Any,
    ) -> None:
        """Descend a ``properties`` object. New/removed property NAMES are governed by
        the field-presence rule against the OLD parent object's ``additionalProperties``
        — NOT by the generic added/removed keyword rules."""
        base = _join(path, "properties")
        if not isinstance(ov, dict) or not isinstance(nv, dict):
            if ov != nv:
                self._flag(base, "properties", ov, nv,
                           "malformed properties change; cannot prove narrowing")
            return

        for name in sorted(set(ov) | set(nv)):
            here = _join(base, name)
            if name in ov and name in nv:
                self.compare(ov[name], nv[name], here)
            elif name in nv:  # new property key
                # Field-presence rule: a new field is a relaxation UNLESS the old parent's
                # additionalProperties was provably open — True, absent (None), or an empty
                # allow-all schema ({}). A False value or a non-empty constraining schema
                # dict must flag (fail-closed): adding an explicit field with a looser schema
                # than the constraint admits inputs the old schema rejected.
                old_ap_open = old_additional is True or old_additional is None or old_additional == {}
                if not old_ap_open:
                    self._flag(
                        here, "properties", None, nv[name],
                        "new field accepted where the old schema did not provably accept it"
                        " (additionalProperties was not open)",
                    )
            else:  # removed property key
                # Removing a field drops its type/shape constraint. Under a closed
                # object (additionalProperties:false) the field becomes forbidden — a
                # provable narrowing (pass). Otherwise the field may still appear,
                # now unconstrained -> wider -> flag.
                if old_additional is not False:
                    self._flag(
                        here, "properties", ov[name], None,
                        "field definition removed; its constraint no longer applies and the field may still appear",
                    )

    def _compare_defs(
        self,
        ov: Any,
        nv: Any,
        path: str,
        keyword: str,
    ) -> None:
        """Descend a ``$defs``/``definitions`` block. Only entries present on BOTH sides
        are compared: a one-sided entry is either a renamed model (its ``$ref`` change is
        flagged at the referring path) or a dead/unreferenced definition (ignored). The
        walk follows the literal dict structure and never resolves ``$ref``, so a
        recursive model cannot cause infinite recursion."""
        base = _join(path, keyword)
        if not isinstance(ov, dict) or not isinstance(nv, dict):
            if ov != nv:
                self._flag(base, keyword, ov, nv,
                           "malformed definitions block change; cannot prove narrowing")
            return
        for name in sorted(set(ov) & set(nv)):
            if ov[name] != nv[name]:
                self.compare(ov[name], nv[name], _join(base, name))

    def _compare_items(self, ov: Any, nv: Any, path: str) -> None:
        """Descend an ``items`` slot (a single sub-schema or a positional tuple)."""
        if isinstance(ov, dict) and isinstance(nv, dict):
            self.compare(ov, nv, path)
            return
        if isinstance(ov, list) and isinstance(nv, list) and len(ov) == len(nv):
            for i, (a, b) in enumerate(zip(ov, nv)):
                if a != b:
                    self.compare(a, b, _join(path, str(i)))
            return
        if ov != nv:
            self._flag(path, "items", ov, nv, "items shape changed; cannot prove narrowing")

    def _compare_additional_properties(self, ov: Any, nv: Any, path: str) -> None:
        """Compare ``additionalProperties``. A rise in permissiveness (False < schema <
        True) is a relaxation; a fall is a tightening; an equal-rank schema pair is
        descended."""
        ro, rn = _ap_rank(ov), _ap_rank(nv)
        if rn > ro:
            self._flag(
                path, "additionalProperties", ov, nv,
                "additionalProperties loosened; previously rejected fields now accepted",
            )
            return
        if rn < ro:
            return  # tightened
        if isinstance(ov, dict) and isinstance(nv, dict):
            self.compare(ov, nv, path)


# ---------------------------------------------------------------------------
# Diff + CI gate
# ---------------------------------------------------------------------------

def schema_relaxation_diff(old: PipelineConfig, new: PipelineConfig) -> SchemaRelaxationDiff:
    """Compare the input schema of every agent present in BOTH manifests (matched by
    name). Added/removed agents are the authority diff's concern, not this check."""
    old_agents = {a.name: a for a in old.agents}
    new_agents = {a.name: a for a in new.agents}
    relaxations: list[SchemaRelaxation] = []
    for name in sorted(set(old_agents) & set(new_agents)):
        _SchemaComparator(name, relaxations).compare(
            old_agents[name].input_schema,
            new_agents[name].input_schema,
            "",
        )
    relaxations.sort(key=lambda r: (r.agent, r.path, r.keyword))
    return SchemaRelaxationDiff(
        old_fingerprint=canonical_fingerprint({a.name: a.input_schema for a in old.agents}),
        new_fingerprint=canonical_fingerprint({a.name: a.input_schema for a in new.agents}),
        relaxations=tuple(relaxations),
    )


def check_no_schema_relaxation(old: PipelineConfig, new: PipelineConfig) -> None:
    """Raise ``SchemaRelaxationError`` if any name-matched agent's INPUT schema was
    relaxed (accepts a strictly wider input set); return None on identity or a pure
    tightening. Fail-closed: any change that cannot be proven a narrowing is a
    relaxation. Admission / CI-time only — never raised by the pipeline runtime."""
    diff = schema_relaxation_diff(old, new)
    if not diff.ok:
        raise SchemaRelaxationError(diff.certificate(), diff=diff)
