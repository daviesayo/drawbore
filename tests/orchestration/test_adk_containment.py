import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "drawbore"

# Any import of the top-level `google` package, in any form:
#   import google
#   import google.adk[.x]
#   import google as g
#   from google import adk
#   from google.adk[.x] import ...
# google.adk is an implementation detail of the ADK adapter and MUST live only
# under drawbore.orchestration — the engine-containment invariant enforced
# project-wide.
_GOOGLE_IMPORT = re.compile(r"^\s*(?:import\s+google\b|from\s+google\b)", re.MULTILINE)


def test_google_is_imported_only_under_orchestration():
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        if rel.parts[0] == "orchestration":
            continue
        if _GOOGLE_IMPORT.search(path.read_text()):
            offenders.append(str(rel))
    assert offenders == [], f"google package imported outside orchestration: {offenders}"
