"""The scoped tool-registry overlay for local test mode.

Builds a fresh ``ToolRegistry`` containing exactly the tools the pipeline's steps
DECLARE — never an undeclared tool (a declared tool was already accepted by
``Pipeline.add``/``from_json``). Each entry preserves the original tool's
name/allowed-operations/schema/kind so the proxy enforces the SAME authority as
production; only the handler is swapped:

- a ``mock_tools`` entry -> the mock handler (still invoked via ``ToolProxy``)
- a ref in ``allow_real_tools`` -> the original handler (explicit opt-in)
- ``evidence://retrieve`` -> rebound to the TEST evidence store
- otherwise -> a fail-closed handler that raises ``TestingError`` on invoke
"""

from __future__ import annotations

from typing import Any, Collection, Mapping

from drawbore.evidence import EVIDENCE_TOOL_REF, EvidenceStore, register_evidence_tool
from drawbore.pipeline.graph import JoinNode
from drawbore.tools import Tool, ToolRegistry, TrustLabel

from .errors import TestingError
from .models import ToolMock, make_tool_handler


def _register_like(overlay: ToolRegistry, ref: str, handler, original: Tool | None) -> None:
    """Register ``handler`` under ``ref`` preserving the original declaration fields
    (allowed_operations/schema/kind/source_trust/exfil_capable). Defaults apply when
    the tool was not in the source registry (shouldn't happen for a declared tool)."""
    allowed = original.allowed_operations if original is not None else ("invoke",)
    schema = original.schema if original is not None else None
    kind = original.kind if original is not None else "custom"
    source_trust = original.source_trust if original is not None else TrustLabel.TRUSTED
    exfil_capable = original.exfil_capable if original is not None else False
    if kind == "builtin":
        overlay.register_builtin(
            ref, handler, allowed_operations=allowed, schema=schema,
            source_trust=source_trust, exfil_capable=exfil_capable,
        )
    elif kind == "mcp":
        overlay.register_mcp_tool(
            ref, handler, allowed_operations=allowed, schema=schema,
            source_trust=source_trust, exfil_capable=exfil_capable,
        )
    else:
        overlay.register_tool(
            ref, handler, allowed_operations=allowed, schema=schema,
            source_trust=source_trust, exfil_capable=exfil_capable,
        )


def _fail_closed_handler(ref: str):
    async def handler(args: Any) -> Any:
        raise TestingError(
            f"tool '{ref}' has no mock and is not in allow_real_tools; local test "
            f"mode will not call the outside world (add a mock_tools entry or list it "
            f"in allow_real_tools)"
        )
    return handler


def build_scoped_registry(
    pipeline,
    *,
    mock_tools: Mapping[str, ToolMock],
    allow_real_tools: Collection[str],
    evidence_store: EvidenceStore,
) -> ToolRegistry:
    """Construct the per-test registry overlay. Fails closed if a mock or
    ``allow_real_tools`` ref is not declared by any step (a typo guard)."""
    declared: set[str] = set()
    for step in pipeline.steps:
        if isinstance(step, JoinNode):
            continue
        declared.update(step.agent.spec.tools)

    for ref in list(mock_tools) + list(allow_real_tools):
        if ref not in declared:
            raise TestingError(
                f"mock/allow_real for tool '{ref}' but no pipeline step declares it"
            )

    source = pipeline._registry
    overlay = ToolRegistry()
    for ref in sorted(declared):
        if ref == EVIDENCE_TOOL_REF:
            # Drawbore-owned policy gate: rebind to the test store. The fresh
            # overlay has no prior binding, so this never hits a duplicate-ref error.
            register_evidence_tool(overlay, store=evidence_store)
            continue
        original = source.get(ref) if source.has(ref) else None
        if ref in mock_tools:
            if original is None:
                raise TestingError(
                    f"mock_tools names '{ref}' but it is not in the pipeline registry"
                )
            _register_like(overlay, ref, make_tool_handler(ref, mock_tools[ref]), original)
        elif ref in allow_real_tools:
            if original is None:
                raise TestingError(
                    f"allow_real_tools names '{ref}' but it is not in the pipeline registry"
                )
            _register_like(overlay, ref, original.handler, original)
        else:
            _register_like(overlay, ref, _fail_closed_handler(ref), original)
    return overlay
