"""A deterministic-only `import drawbore` must not pay for provider/engine
SDKs. Subprocess-isolated — in this process, other test modules already imported
litellm, so an in-process assertion would be vacuous."""

import subprocess
import sys


def test_import_drawbore_pulls_no_provider_sdks():
    code = (
        "import sys; import drawbore; "
        "banned = [m for m in ('litellm', 'google.adk') if m in sys.modules]; "
        "assert not banned, f'eagerly imported: {banned}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_lazy_symbols_still_importable():
    code = (
        "from drawbore.orchestration import ADKEngine, make_scripted_model_factory; "
        "assert ADKEngine.__name__ == 'ADKEngine'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
