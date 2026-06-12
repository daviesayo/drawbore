# tests/ratchet/test_ratchet_containment.py
"""The ratchet's import boundary, enforced.

The ratchet composes config + testing + pipeline + state + schema + escalation.
It must never import an engine, provider, or transport SDK — and no module outside
the ratchet may import it.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "drawbore"

FORBIDDEN_IN_RATCHET = ("google.adk", "litellm", "mcp", "opentelemetry")

def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_ratchet_imports_no_engine_provider_or_transport_sdk():
    for path in (SRC / "ratchet").rglob("*.py"):
        for mod in _imports(path):
            for forbidden in FORBIDDEN_IN_RATCHET:
                assert mod != forbidden and not mod.startswith(forbidden + "."), (
                    f"{path.name} imports forbidden module {mod!r}"
                )


def test_no_module_outside_the_ratchet_imports_it():
    for path in SRC.rglob("*.py"):
        if "ratchet" in path.relative_to(SRC).parts:
            continue
        for mod in _imports(path):
            assert "ratchet" not in mod.split("."), (
                f"{path.relative_to(SRC)} imports {mod!r} — non-ratchet modules "
                f"must not import the ratchet"
            )
