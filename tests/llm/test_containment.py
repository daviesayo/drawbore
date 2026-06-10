import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "drawbore"


def _files(pkg):
    return list((SRC / pkg).rglob("*.py"))


def _top_package(path: pathlib.Path) -> str:
    # First path component INSIDE src/drawbore (e.g. "llm", "orchestration").
    # Anchored on SRC rather than parts.index("drawbore") so a checkout whose repo
    # directory is itself named "drawbore" does not collide.
    return path.relative_to(SRC).parts[0]


def test_litellm_imported_only_under_llm():
    offenders = []
    for path in SRC.rglob("*.py"):
        if _top_package(path) == "llm":
            continue
        text = path.read_text()
        if "import litellm" in text or "from litellm" in text:
            offenders.append(str(path))
    assert offenders == [], f"litellm imported outside drawbore.llm: {offenders}"


def test_llm_does_not_import_orchestration_or_higher():
    forbidden = ("drawbore.orchestration", "drawbore.pipeline", "drawbore.audit",
                 "drawbore.observability", "drawbore.agent", "drawbore.testing")
    files = _files("llm")
    assert files, "no llm files found (vacuous guard)"
    offenders = []
    for path in files:
        text = path.read_text()
        for mod in forbidden:
            if f"import {mod}" in text or f"from {mod}" in text:
                offenders.append((str(path), mod))
    assert offenders == [], f"drawbore.llm must stay a near-leaf: {offenders}"


def test_google_adk_still_only_under_orchestration():
    offenders = []
    for path in SRC.rglob("*.py"):
        if _top_package(path) == "orchestration":
            continue
        text = path.read_text()
        if "import google.adk" in text or "from google.adk" in text or "from google import adk" in text:
            offenders.append(str(path))
    assert offenders == [], f"google.adk imported outside drawbore.orchestration: {offenders}"
