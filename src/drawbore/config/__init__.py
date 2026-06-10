"""JSON config layer — the source of truth for a pipeline definition. A manifest
plus a fail-closed resolver: ``to_*`` emit a typed manifest of a pipeline's
contract + policy; ``from_*`` resolve symbolic agent refs to live objects and
rebuild the pipeline, failing closed on drift/missing/invalid.
"""

from .authority import (
    AuthorityDiff,
    CapabilityFact,
    CapabilityFootprint,
    authority_diff,
    check_no_new_authority,
    effective_authority,
)
from .catalog import AgentCatalog
from .errors import AuthorityRegressionError, ConfigResolutionError
from .models import PipelineConfig
from .resolver import from_config, from_json
from .serialization import to_config, to_json

__all__ = [
    "AuthorityDiff",
    "AuthorityRegressionError",
    "CapabilityFact",
    "CapabilityFootprint",
    "AgentCatalog",
    "ConfigResolutionError",
    "PipelineConfig",
    "authority_diff",
    "check_no_new_authority",
    "effective_authority",
    "from_config",
    "from_json",
    "to_config",
    "to_json",
]
