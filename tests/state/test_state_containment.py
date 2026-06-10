"""Containment guard for drawbore.state.

``drawbore.state`` is a low-level primitive — checkpoint storage and per-run
message passing. It may only import stdlib, pydantic, and ``drawbore.tools.taint``
(the TrustLabel type it now persists). It must never pull in higher Drawbore
subsystems (pipeline, orchestration, llm, testing, mcp, audit, observability,
evidence, escalation, identity, versioning, config, agent) or third-party
provider SDKs (google/ADK, litellm, mcp SDK, opentelemetry).
"""

import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "drawbore"

# Third-party provider SDKs that must never appear.
_GOOGLE = re.compile(r"^\s*(?:import\s+google\b|from\s+google\b)", re.MULTILINE)
_LITELLM = re.compile(r"^\s*(?:import\s+litellm\b|from\s+litellm\b)", re.MULTILINE)
_MCP_SDK = re.compile(r"^\s*(?:import\s+mcp\b|from\s+mcp\b)", re.MULTILINE)
_OTEL = re.compile(r"^\s*(?:import\s+opentelemetry\b|from\s+opentelemetry\b)", re.MULTILINE)

# Higher Drawbore subsystems that state must not depend on.
_FORBIDDEN_DRAWBORE = re.compile(
    r"^\s*from\s+drawbore\."
    r"(pipeline|orchestration|llm|testing|mcp|audit|observability|"
    r"evidence|escalation|identity|versioning|config|agent)\b",
    re.MULTILINE,
)


def _files() -> list[pathlib.Path]:
    return list((SRC / "state").rglob("*.py"))


def test_state_containment_is_not_vacuous() -> None:
    # Guard: a refactor that moves/renames the package must not silently neuter the
    # containment assertions below.
    assert _files(), (
        f"no .py files found under {SRC / 'state'}; containment tests would be vacuous"
    )


def test_state_imports_no_provider_sdk() -> None:
    offenders = [
        str(p.relative_to(SRC))
        for p in _files()
        if _GOOGLE.search(p.read_text())
        or _LITELLM.search(p.read_text())
        or _MCP_SDK.search(p.read_text())
        or _OTEL.search(p.read_text())
    ]
    assert offenders == [], (
        "drawbore.state must not import google/ADK, litellm, the mcp SDK, "
        f"or opentelemetry: {offenders}"
    )


def test_state_does_not_import_higher_drawbore_subsystems() -> None:
    offenders = [
        str(p.relative_to(SRC))
        for p in _files()
        if _FORBIDDEN_DRAWBORE.search(p.read_text())
    ]
    assert offenders == [], (
        "drawbore.state must not import pipeline/orchestration/llm/testing/mcp/"
        "audit/observability/evidence/escalation/identity/versioning/config/agent: "
        f"{offenders}"
    )


# ---------------------------------------------------------------------------
# Per-module containment: step_seal, resume_ledger, and _canon
# ---------------------------------------------------------------------------

# Patterns for higher subsystems that step_seal and resume_ledger must not
# import at RUNTIME (TYPE_CHECKING-guarded imports are source-only and are
# allowed — they never appear as bare `from drawbore.X import` at module level
# outside the TYPE_CHECKING block, so a source scan is the right mechanism).
_STEP_SEAL_FORBIDDEN = re.compile(
    r"^\s*from\s+drawbore\.(config|agent|evidence|pipeline|orchestration|"
    r"llm|testing|mcp|audit|observability|escalation|identity|versioning)\b",
    re.MULTILINE,
)
_CANON_FORBIDDEN_THIRD_PARTY = re.compile(
    r"^\s*(?:import\s+pydantic\b|from\s+pydantic\b"
    r"|import\s+drawbore\b|from\s+drawbore\b)",
    re.MULTILINE,
)


def _step_seal_file() -> pathlib.Path:
    return SRC / "state" / "step_seal.py"


def _resume_ledger_file() -> pathlib.Path:
    return SRC / "state" / "resume_ledger.py"


def _canon_file() -> pathlib.Path:
    return SRC / "_canon.py"


def test_step_seal_no_runtime_imports_of_higher_subsystems() -> None:
    """drawbore.state.step_seal must not import config/agent/evidence/google.adk/
    mcp/litellm at RUNTIME (TYPE_CHECKING-only is allowed)."""
    p = _step_seal_file()
    assert p.exists(), f"step_seal.py not found at {p}"
    src = p.read_text()
    assert not _STEP_SEAL_FORBIDDEN.search(src), (
        "drawbore.state.step_seal has a runtime import of a forbidden subsystem; "
        "use TYPE_CHECKING guard for type-annotation-only imports"
    )
    for pattern, label in [
        (_GOOGLE, "google/ADK"),
        (_LITELLM, "litellm"),
        (_MCP_SDK, "mcp SDK"),
    ]:
        assert not pattern.search(src), (
            f"drawbore.state.step_seal must not import {label}"
        )


def test_resume_ledger_no_runtime_imports_of_higher_subsystems() -> None:
    """drawbore.state.resume_ledger must not import config/agent/evidence/
    google.adk/mcp/litellm at RUNTIME."""
    p = _resume_ledger_file()
    assert p.exists(), f"resume_ledger.py not found at {p}"
    src = p.read_text()
    assert not _STEP_SEAL_FORBIDDEN.search(src), (
        "drawbore.state.resume_ledger has a runtime import of a forbidden subsystem"
    )
    for pattern, label in [
        (_GOOGLE, "google/ADK"),
        (_LITELLM, "litellm"),
        (_MCP_SDK, "mcp SDK"),
    ]:
        assert not pattern.search(src), (
            f"drawbore.state.resume_ledger must not import {label}"
        )


def test_canon_imports_stdlib_only() -> None:
    """drawbore._canon must import stdlib only — no Pydantic, no drawbore subsystems."""
    p = _canon_file()
    assert p.exists(), f"_canon.py not found at {p}"
    src = p.read_text()
    # Allow the module's own name in comments/docstrings but not as an import.
    assert not _CANON_FORBIDDEN_THIRD_PARTY.search(src), (
        "drawbore._canon must not import pydantic or any drawbore subsystem; "
        "it is a stdlib-only leaf shared across subsystems"
    )
