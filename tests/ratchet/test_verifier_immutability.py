# tests/ratchet/test_verifier_immutability.py
"""The proposal surface cannot reach the verifier.

A manifest is data: no field is resolved through Python import machinery, an
unknown agent ref fails closed instead of being treated as an import path, and
the config + ratchet packages contain no dynamic-import escape hatch.
"""
import ast
import json
import pathlib

import pytest

from drawbore.config import AgentCatalog, from_json
from drawbore.config.errors import ConfigResolutionError

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "drawbore"


def test_unknown_agent_ref_fails_closed_never_imported():
    # a ref that LOOKS like an import path must be treated as an unknown ref,
    # never resolved through import machinery.
    bad = {
        "schema_version": 1,
        "pipeline": {"name": "p", "version": "1.0.0"},
        "agents": [{
            "name": "evil", "ref": "os.path:join", "version": "1.0.0",
            "risk_tier": "low", "requires_human_approval": False,
            "context_access": "none", "tools": [], "model": None,
            "fallback_model": None, "instructions": None,
            "input_schema": {}, "output_schema": {},
            "input_schema_hash": "sha256:" + "0" * 64,
            "output_schema_hash": "sha256:" + "0" * 64,
        }],
        "steps": [{"agent": "evil", "input": {"source": "initial"}}],
    }
    with pytest.raises(ConfigResolutionError, match="not registered"):
        from_json(json.dumps(bad), agents=AgentCatalog())


def test_config_and_ratchet_use_no_dynamic_import_machinery():
    for package in ("config", "ratchet"):
        for path in (SRC / package).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mods = (
                        [a.name for a in node.names] if isinstance(node, ast.Import)
                        else [node.module or ""]
                    )
                    assert not any(
                        m == "importlib" or m.startswith("importlib.")
                        for m in mods
                    ), f"{package}/{path.name} imports importlib"
                if isinstance(node, ast.Call):
                    fn = node.func
                    name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                    assert name not in {"eval", "exec", "__import__"}, (
                        f"{package}/{path.name} calls {name}()"
                    )
