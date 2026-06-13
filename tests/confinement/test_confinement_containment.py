# tests/confinement/test_confinement_containment.py
"""The confinement module's import boundary, enforced.

The confinement module composes only ``_canon``, the config layer's fingerprint
helpers, Pydantic, and stdlib. It must never import an engine, provider, or
transport SDK — and no module outside the confinement package may import it,
except the pipeline package (which mints the receipt in its ``run()`` finally
block).
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "drawbore"

FORBIDDEN_IN_CONFINEMENT = ("google.adk", "litellm", "mcp", "opentelemetry")


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_confinement_imports_no_engine_provider_or_transport_sdk():
    for path in (SRC / "confinement").rglob("*.py"):
        for mod in _imports(path):
            for forbidden in FORBIDDEN_IN_CONFINEMENT:
                assert mod != forbidden and not mod.startswith(forbidden + "."), (
                    f"{path.name} imports forbidden module {mod!r}"
                )


def test_no_module_outside_confinement_imports_it_except_pipeline():
    for path in SRC.rglob("*.py"):
        parts = path.relative_to(SRC).parts
        if "confinement" in parts:
            continue
        if "pipeline" in parts:
            continue
        for mod in _imports(path):
            assert "confinement" not in mod.split("."), (
                f"{path.relative_to(SRC)} imports {mod!r} — only the pipeline "
                f"package may import drawbore.confinement"
            )
