import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "drawbore"

_GOOGLE = re.compile(r"^\s*(?:import\s+google\b|from\s+google\b)", re.MULTILINE)
_LITELLM = re.compile(r"^\s*(?:import\s+litellm\b|from\s+litellm\b)", re.MULTILINE)
_MCP = re.compile(r"^\s*(?:import\s+mcp\b|from\s+mcp\b)", re.MULTILINE)


def _files():
    return list((SRC / "testing").rglob("*.py"))


def test_testing_containment_is_not_vacuous():
    # Guard: a refactor that moves/renames the package must not silently neuter the
    # containment assertion below.
    assert _files(), f"no .py files found under {SRC / 'testing'}; containment test would be vacuous"


def test_testing_imports_no_provider_sdk():
    offenders = [
        str(p.relative_to(SRC))
        for p in _files()
        if _GOOGLE.search(p.read_text())
        or _LITELLM.search(p.read_text())
        or _MCP.search(p.read_text())
    ]
    assert offenders == [], (
        f"drawbore.testing must not import google/litellm/the mcp SDK — the fake "
        f"loop model is a BaseLlm and lives under drawbore.orchestration: {offenders}"
    )
