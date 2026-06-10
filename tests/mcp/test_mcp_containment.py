import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "drawbore"

# Match an import of the top-level `mcp` SDK package (not `drawbore.mcp`).
_MCP_SDK_IMPORT = re.compile(r"^\s*(?:import\s+mcp\b|from\s+mcp\b)", re.MULTILINE)
_GOOGLE_IMPORT = re.compile(r"^\s*(?:import\s+google\b|from\s+google\b)", re.MULTILINE)


def test_mcp_sdk_is_imported_only_under_drawbore_mcp():
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        if rel.parts[0] == "mcp":
            continue
        if _MCP_SDK_IMPORT.search(path.read_text()):
            offenders.append(str(rel))
    assert offenders == [], f"`mcp` SDK imported outside drawbore/mcp: {offenders}"


def test_drawbore_mcp_does_not_import_google():
    # The MCP integration does not touch ADK; the live transport uses the `mcp`
    # SDK directly (pulled via google-adk[mcp], but `mcp` != `google.adk`).
    offenders = []
    for path in (SRC / "mcp").rglob("*.py"):
        if _GOOGLE_IMPORT.search(path.read_text()):
            offenders.append(str(path.relative_to(SRC)))
    assert offenders == [], f"drawbore.mcp must not import google.adk: {offenders}"
