import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "drawbore"

_GOOGLE = re.compile(r"^\s*(?:import\s+google\b|from\s+google\b)", re.MULTILINE)
_LITELLM = re.compile(r"^\s*(?:import\s+litellm\b|from\s+litellm\b)", re.MULTILINE)
_MCP = re.compile(r"^\s*(?:import\s+mcp\b|from\s+mcp\b)", re.MULTILINE)
# config is a construction layer: it must not pull in engines/gateways/stores/exporters.
# `observability` is forbidden too (matching the sibling evidence containment test):
# config computes its OWN schema_fingerprint and must never reach for the OTel
# exporter surface or `observability.payload_hash`.
_FORBIDDEN = re.compile(
    r"^\s*from\s+drawbore\.(orchestration|llm|mcp|audit|observability)\b",
    re.MULTILINE,
)


def _files():
    return list((SRC / "config").rglob("*.py"))


def test_config_containment_is_not_vacuous():
    # Guard: if a refactor renames/moves the config package, _files() would return []
    # and BOTH containment tests below would pass trivially — silently neutering a
    # non-negotiable invariant. Fail loudly instead.
    assert _files(), f"no .py files found under {SRC / 'config'}; containment tests would be vacuous"


def test_config_imports_no_adk_litellm_or_mcp_sdk():
    offenders = [
        str(p.relative_to(SRC))
        for p in _files()
        if _GOOGLE.search(p.read_text())
        or _LITELLM.search(p.read_text())
        or _MCP.search(p.read_text())
    ]
    assert offenders == [], f"config must not import google.adk / litellm / the mcp SDK: {offenders}"


def test_config_does_not_import_runtime_subsystems():
    offenders = [str(p.relative_to(SRC)) for p in _files() if _FORBIDDEN.search(p.read_text())]
    assert offenders == [], f"config must not import orchestration/llm/mcp/audit/observability: {offenders}"
