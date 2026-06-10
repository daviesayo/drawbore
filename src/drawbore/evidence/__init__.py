"""Evidence compression layer. A Drawbore extension. Near-leaf: imports only
stdlib + pydantic + drawbore.errors; no ADK, no mcp, no
agent/pipeline/tools/orchestration/llm/audit/observability."""

from .compress import compress_for_model
from .errors import (
    EvidenceError,
    EvidenceRetrievalError,
    EvidenceStoreError,
    EvidenceTransformError,
)
from .policy import EvidencePolicy
from .records import EvidenceDecision, EvidenceHandle
from .retrieval import EVIDENCE_TOOL_REF, register_evidence_tool
from .store import EvidenceStore, InMemoryEvidenceStore
from .tokens import estimate_tokens
from .transforms import EvidenceTransform, get_transform, json_rows, logs, register_transform

__all__ = [
    "EvidenceError",
    "EvidenceStoreError",
    "EvidenceTransformError",
    "EvidenceRetrievalError",
    "EvidencePolicy",
    "EvidenceDecision",
    "EvidenceHandle",
    "EvidenceStore",
    "InMemoryEvidenceStore",
    "estimate_tokens",
    "EvidenceTransform",
    "get_transform",
    "register_transform",
    "json_rows",
    "logs",
    "compress_for_model",
    "EVIDENCE_TOOL_REF",
    "register_evidence_tool",
]
