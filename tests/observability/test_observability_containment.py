import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "drawbore"

_GOOGLE = re.compile(r"^\s*(?:import\s+google\b|from\s+google\b)", re.MULTILINE)
_MCP = re.compile(r"^\s*(?:import\s+mcp\b|from\s+mcp\b)", re.MULTILINE)
# observability is a leaf: it must not import these drawbore subsystems.
_FORBIDDEN_DEPS = re.compile(
    r"^\s*from\s+drawbore\.(agent|pipeline|tools|orchestration|llm|mcp)\b", re.MULTILINE
)


def _files(pkg):
    return list((SRC / pkg).rglob("*.py"))


def test_observability_and_audit_import_no_adk_or_mcp_sdk():
    offenders = []
    for pkg in ("observability", "audit"):
        for path in _files(pkg):
            text = path.read_text()
            if _GOOGLE.search(text) or _MCP.search(text):
                offenders.append(str(path.relative_to(SRC)))
    assert offenders == [], f"observability/audit must not import google.adk or the mcp SDK: {offenders}"


def test_observability_is_a_leaf_no_subsystem_imports():
    offenders = [
        str(p.relative_to(SRC)) for p in _files("observability") if _FORBIDDEN_DEPS.search(p.read_text())
    ]
    assert offenders == [], f"observability must not import agent/pipeline/tools/orchestration/llm/mcp: {offenders}"
