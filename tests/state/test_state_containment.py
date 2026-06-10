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
