"""Packaging surface: the wheel must carry type info and clean metadata."""

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _build_wheel(tmp_path):
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=ROOT, check=True, capture_output=True,
    )
    return next(tmp_path.glob("drawbore-*.whl"))


def test_wheel_ships_py_typed_and_excludes_tests(tmp_path):
    wheel = _build_wheel(tmp_path)
    names = zipfile.ZipFile(wheel).namelist()
    assert "drawbore/py.typed" in names
    assert not [n for n in names if n.startswith(("tests/", "examples/", "docs/"))]


def test_pyproject_release_metadata():
    import tomllib

    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    proj = cfg["project"]
    assert proj["urls"]["Repository"] == "https://github.com/daviesayo/drawbore"
    assert any(c.startswith("Development Status") for c in proj["classifiers"])
    assert "dev" not in proj.get("optional-dependencies", {}), "dev deps belong in [dependency-groups]"
    assert "dependency-groups" in cfg
