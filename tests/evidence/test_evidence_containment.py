import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "drawbore"

_GOOGLE = re.compile(r"^\s*(?:import\s+google\b|from\s+google\b)", re.MULTILINE)
_MCP = re.compile(r"^\s*(?:import\s+mcp\b|from\s+mcp\b)", re.MULTILINE)
# evidence is a near-leaf: it must not import these drawbore subsystems.
_FORBIDDEN = re.compile(
    r"^\s*from\s+drawbore\.(agent|pipeline|tools|orchestration|llm|mcp|audit|observability)\b",
    re.MULTILINE,
)


def _files():
    return list((SRC / "evidence").rglob("*.py"))


def test_evidence_imports_no_adk_or_mcp_sdk():
    offenders = [str(p.relative_to(SRC)) for p in _files() if _GOOGLE.search(p.read_text()) or _MCP.search(p.read_text())]
    assert offenders == [], f"evidence must not import google.adk or the mcp SDK: {offenders}"


def test_evidence_is_a_near_leaf():
    offenders = [str(p.relative_to(SRC)) for p in _files() if _FORBIDDEN.search(p.read_text())]
    assert offenders == [], f"evidence must not import agent/pipeline/tools/orchestration/llm/mcp/audit/observability: {offenders}"
