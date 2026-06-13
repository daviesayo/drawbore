"""Tool registry.

Tools are registered with a name, an async handler, the operations they permit,
and an optional Pydantic schema for their arguments. ``kind`` records provenance
(custom vs built-in).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, ClassVar
from types import MappingProxyType

from ..errors import ToolAccessError
from ..taint import TrustLabel

ToolHandler = Callable[[Any], Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    handler: ToolHandler
    allowed_operations: tuple[str, ...] = ("invoke",)
    schema: Any = None
    kind: str = "custom"
    source_trust: TrustLabel = TrustLabel.TRUSTED   # trust of the data this tool returns; the TRUSTED
                                                    # default neither taints a step nor gets gated —
                                                    # declare UNTRUSTED (or use register_mcp_tool) for
                                                    # external sources
    exfil_capable: bool = False                     # True if invoking it can move data outward across
                                                    # the trust boundary; such calls are refused while
                                                    # a step's taint scope is UNTRUSTED
    effectful: bool = True                          # False for pure-read tools that must NOT be
                                                    # ledgered or replayed on resume (e.g. evidence
                                                    # retrieval); True for any tool that may produce
                                                    # a durable side-effect

    _VALID_KINDS: ClassVar[frozenset[str]] = frozenset({"custom", "builtin", "mcp"})

    def __post_init__(self) -> None:
        # Validate kind before anything else — fail closed on unknown provenance.
        if self.kind not in self._VALID_KINDS:
            raise ValueError(
                f"Tool kind must be one of {sorted(self._VALID_KINDS)!r}; got {self.kind!r}"
            )
        # Coerce so the proxy's identity checks can't be bypassed by a raw string
        # (a string that isn't a valid TrustLabel raises ValueError here, fail closed).
        if not isinstance(self.source_trust, TrustLabel):
            object.__setattr__(self, "source_trust", TrustLabel(self.source_trust))
        if not isinstance(self.exfil_capable, bool):
            object.__setattr__(self, "exfil_capable", bool(self.exfil_capable))
        if not isinstance(self.effectful, bool):
            object.__setattr__(self, "effectful", bool(self.effectful))


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register_tool(
        self,
        name: str,
        handler: ToolHandler,
        allowed_operations=("invoke",),
        schema: Any = None,
        *,
        source_trust: TrustLabel = TrustLabel.TRUSTED,
        exfil_capable: bool = False,
        effectful: bool = True,
    ) -> Tool:
        return self._register(
            name, handler, allowed_operations, schema,
            kind="custom",
            source_trust=source_trust,
            exfil_capable=exfil_capable,
            effectful=effectful,
        )

    def register_builtin(
        self,
        name: str,
        handler: ToolHandler,
        allowed_operations=("invoke",),
        schema: Any = None,
        *,
        source_trust: TrustLabel = TrustLabel.TRUSTED,
        exfil_capable: bool = False,
        effectful: bool = True,
    ) -> Tool:
        return self._register(
            name, handler, allowed_operations, schema,
            kind="builtin",
            source_trust=source_trust,
            exfil_capable=exfil_capable,
            effectful=effectful,
        )

    def register_mcp_tool(
        self,
        name: str,
        handler: ToolHandler,
        allowed_operations=("invoke",),
        schema: Any = None,
        *,
        source_trust: TrustLabel = TrustLabel.UNTRUSTED,
        exfil_capable: bool = False,
        effectful: bool = True,
    ) -> Tool:
        """Register a tool backed by an MCP server (``kind="mcp"``). The handler is
        an opaque async callable (built by ``drawbore.mcp``); the registry stays
        MCP-agnostic and imports nothing from ``drawbore.mcp`` (layering invariant).
        Untrusted-source by default (external by nature)."""
        return self._register(
            name, handler, allowed_operations, schema,
            kind="mcp",
            source_trust=source_trust,
            exfil_capable=exfil_capable,
            effectful=effectful,
        )

    def _register(
        self,
        name: str,
        handler: ToolHandler,
        allowed_operations,
        schema: Any,
        kind: str,
        *,
        source_trust: TrustLabel = TrustLabel.TRUSTED,
        exfil_capable: bool = False,
        effectful: bool = True,
    ) -> Tool:
        if name in self._tools:
            raise ValueError(f"tool '{name}' is already registered")
        tool = Tool(
            name=name,
            handler=handler,
            allowed_operations=tuple(allowed_operations),
            schema=schema,
            kind=kind,
            source_trust=source_trust,
            exfil_capable=exfil_capable,
            effectful=effectful,
        )
        self._tools[name] = tool
        return tool

    def tools(self) -> Mapping[str, Tool]:
        """Return a read-only view of all registered tools."""
        return MappingProxyType(self._tools)

    def clone(self) -> "ToolRegistry":
        """Return a new ToolRegistry whose tool set is a shallow copy of this one.

        Frozen ``Tool`` objects are safe to share. Registration side-effects
        (e.g. callbacks) are NOT re-run; the copy simply inherits the same
        ``Tool`` instances.
        """
        new = ToolRegistry()
        new._tools = dict(self._tools)
        return new

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolAccessError(f"tool '{name}' is not registered")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools


registry = ToolRegistry()
