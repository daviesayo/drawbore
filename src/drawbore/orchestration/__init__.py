"""Orchestration engine seam.

``ADKEngine`` and ``make_scripted_model_factory`` are deferred via PEP 562: they
transitively import the engine and provider SDKs (hundreds of ms), which a
deterministic-only pipeline never needs. First attribute access pays the one-time
import cost; the result is cached into module globals.
"""

from typing import TYPE_CHECKING

from .engine import OrchestratorEngine, StepExecution, ToolLoopBundle
from .errors import EngineError
from .local import LocalEngine

if TYPE_CHECKING:
    from .adk_engine import ADKEngine
    from .scripted_model import make_scripted_model_factory

__all__ = [
    "OrchestratorEngine",
    "LocalEngine",
    "ADKEngine",
    "StepExecution",
    "ToolLoopBundle",
    "EngineError",
    "make_scripted_model_factory",
]

_LAZY = {
    "ADKEngine": ".adk_engine",
    "make_scripted_model_factory": ".scripted_model",
}


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value  # cache: subsequent access skips __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
