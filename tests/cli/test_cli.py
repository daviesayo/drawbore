"""Tests for the ``drawbore`` CLI (``drawbore.cli:main``).

Exercises the public surface via ``main(argv)`` and asserts exit codes (the CI
contract) and key output. Manifests are generated with the real config layer
(``to_json``) so fixtures are always valid; failures are induced by tampering a
manifest field or by building a genuinely widened/relaxed candidate.
"""

import asyncio
import json
import sys
import types

from pydantic import BaseModel, Field

from drawbore import Pipeline, agent
from drawbore.cli import EXIT_CHECK_FAILED, EXIT_OK, EXIT_USAGE, main
from drawbore.config import AgentCatalog, to_json
from drawbore.tools import ToolRegistry


class _In(BaseModel):
    amount: int


class _Out(BaseModel):
    amount: int


def _make_agent(name="solver", *, version="1.0.0", model=None, tools=None, input_model=_In):
    @agent(
        name=name,
        input=input_model,
        output=_Out,
        version=version,
        model=model,
        tools=tools or [],
    )
    async def fn(value):  # type: ignore[no-untyped-def]
        return _Out(amount=getattr(value, "amount", 0))

    return fn


def _manifest(agent_obj, ref="m:solver", *, registry=None) -> str:
    cat = AgentCatalog()
    cat.register(ref, agent_obj)
    pipeline = Pipeline(name="t") if registry is None else Pipeline(name="t", registry=registry)
    pipeline.add(agent_obj)
    return to_json(pipeline, agents=cat)


def _write(tmp_path, name, text) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# --- structural validation -------------------------------------------------

def test_check_structural_ok(tmp_path, capsys):
    manifest = _write(tmp_path, "m.json", _manifest(_make_agent()))
    assert main(["check", manifest]) == EXIT_OK
    assert "manifest OK" in capsys.readouterr().out


def test_check_bad_json(tmp_path, capsys):
    bad = _write(tmp_path, "bad.json", "{ not valid json")
    assert main(["check", bad]) == EXIT_CHECK_FAILED
    assert "not valid JSON" in capsys.readouterr().err


def test_check_unknown_field_fails_closed(tmp_path, capsys):
    data = json.loads(_manifest(_make_agent()))
    data["surprise"] = True  # extra="forbid" at the top level
    path = _write(tmp_path, "x.json", json.dumps(data))
    assert main(["check", path]) == EXIT_CHECK_FAILED
    assert "not a valid manifest" in capsys.readouterr().err


def test_check_missing_file(capsys):
    assert main(["check", "/no/such/manifest.json"]) == EXIT_USAGE
    assert "could not read" in capsys.readouterr().err


# --- resolution against live code -----------------------------------------

def test_check_resolution_ok_and_drift(tmp_path, monkeypatch, capsys):
    a = _make_agent("solver", version="1.0.0")
    cat = AgentCatalog()
    cat.register("m:solver", a)
    pipeline = Pipeline(name="t")
    pipeline.add(a)
    manifest = to_json(pipeline, agents=cat)

    mod = types.ModuleType("fix_pipeline_cli")
    mod.catalog = cat  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fix_pipeline_cli", mod)

    ok = _write(tmp_path, "ok.json", manifest)
    assert main(["check", ok, "--catalog", "fix_pipeline_cli:catalog"]) == EXIT_OK
    assert "resolution OK" in capsys.readouterr().out

    # Drift: the manifest claims a different version than the resolved agent.
    drifted = json.loads(manifest)
    drifted["agents"][0]["version"] = "9.9.9"
    bad = _write(tmp_path, "drift.json", json.dumps(drifted))
    assert main(["check", bad, "--catalog", "fix_pipeline_cli:catalog"]) == EXIT_CHECK_FAILED
    assert "resolution FAILED" in capsys.readouterr().err


def test_check_bad_catalog_spec(tmp_path, capsys):
    manifest = _write(tmp_path, "m.json", _manifest(_make_agent()))
    assert main(["check", manifest, "--catalog", "not_a_module_attr"]) == EXIT_USAGE
    assert "module:attr" in capsys.readouterr().err


# --- ratchet against a baseline -------------------------------------------

def test_check_against_identity_passes(tmp_path, capsys):
    manifest = _manifest(_make_agent())
    cand = _write(tmp_path, "cand.json", manifest)
    base = _write(tmp_path, "base.json", manifest)
    assert main(["check", cand, "--against", base]) == EXIT_OK
    out = capsys.readouterr().out
    assert "authority OK" in out and "input-schema OK" in out


def test_check_against_authority_widening_fails(tmp_path, capsys):
    base = _write(tmp_path, "base.json", _manifest(_make_agent("solver")))
    reg = ToolRegistry()
    reg.register_tool("risk.lookup", lambda args: {}, allowed_operations=("invoke",))
    widened = _make_agent("solver", model="gpt-4o", tools=["risk.lookup"])
    cand = _write(tmp_path, "cand.json", _manifest(widened, registry=reg))
    assert main(["check", cand, "--against", base]) == EXIT_CHECK_FAILED
    assert "Authority regression check: FAILED" in capsys.readouterr().err


def test_check_against_schema_relaxation_fails(tmp_path, capsys):
    class _Tight(BaseModel):
        name: str = Field(max_length=10)

    class _Loose(BaseModel):
        name: str = Field(max_length=20)

    base = _write(tmp_path, "base.json", _manifest(_make_agent("screen", input_model=_Tight), "m:screen"))
    cand = _write(tmp_path, "cand.json", _manifest(_make_agent("screen", input_model=_Loose), "m:screen"))
    assert main(["check", cand, "--against", base]) == EXIT_CHECK_FAILED
    assert "Input-schema relaxation check: FAILED" in capsys.readouterr().err


# --- new (richer scaffold) -------------------------------------------------

def test_new_scaffold_writes_runnable_package(tmp_path, monkeypatch):
    assert main(["new", "scaffolddemo", "--dir", str(tmp_path)]) == EXIT_OK
    pkg = tmp_path / "scaffolddemo"
    for name in ("__init__.py", "pipeline.py", "test_scaffolddemo.py", "README.md"):
        assert (pkg / name).exists(), name

    # The scaffolded pipeline actually runs against the real drawbore runtime.
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        import importlib

        mod = importlib.import_module("scaffolddemo.pipeline")
        result = asyncio.run(mod.pipeline.run(mod.Input(cents=2500)))
        assert result.status == "completed"
        assert result.outputs["normalize"] == mod.Output(dollars=25.0)
    finally:
        for name in list(sys.modules):
            if name == "scaffolddemo" or name.startswith("scaffolddemo."):
                del sys.modules[name]


def test_new_refuses_overwrite(tmp_path, capsys):
    assert main(["new", "demo", "--dir", str(tmp_path)]) == EXIT_OK
    capsys.readouterr()
    assert main(["new", "demo", "--dir", str(tmp_path)]) == EXIT_USAGE
    assert "refusing to overwrite" in capsys.readouterr().err


def test_new_rejects_invalid_identifier(tmp_path, capsys):
    assert main(["new", "9bad-name", "--dir", str(tmp_path)]) == EXIT_USAGE
    assert "not a valid Python identifier" in capsys.readouterr().err


# --- version / no command --------------------------------------------------

def test_version_prints(capsys):
    assert main(["version"]) == EXIT_OK
    assert "drawbore" in capsys.readouterr().out


def test_no_command_is_usage_error(capsys):
    assert main([]) == EXIT_USAGE
