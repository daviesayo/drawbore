"""LLM gateway — the model boundary. No ADK here."""

from .attempts import ModelAttemptAudit, ModelAudit
from .build import build_model_request
from .chain import resolve_model_chain
from .classify import ProviderOutcome, classify_provider_exception
from .config import (
    FallbackReason,
    JsonValue,
    LLMRuntimeConfig,
    ModelProfile,
    ModelTarget,
    ProviderConfig,
)
from .credentials import CredentialChecker, EnvCredentialChecker
from .errors import LLMConfigError, LLMError, ModelUnavailableError, content_excerpt
from .gateway import LLMGateway, LiteLLMGateway
from .production import ProductionLLMGateway
from .request import ModelRequest, ModelResponse
from .resolution import ModelAttempt, ResolvedModelChain, resolve_chain
from .runtime import LLMRuntime
from .usage import TokenUsage, extract_cost, extract_usage

__all__ = [
    "ModelRequest",
    "ModelResponse",
    "resolve_model_chain",
    "build_model_request",
    "LLMGateway",
    "LiteLLMGateway",
    "ProductionLLMGateway",
    "LLMError",
    "LLMConfigError",
    "ModelUnavailableError",
    "content_excerpt",
    "LLMRuntimeConfig",
    "ModelProfile",
    "ModelTarget",
    "ProviderConfig",
    "LLMRuntime",
    "FallbackReason",
    "JsonValue",
    "CredentialChecker",
    "EnvCredentialChecker",
    "ModelAttempt",
    "ResolvedModelChain",
    "resolve_chain",
    "ModelAttemptAudit",
    "ModelAudit",
    "ProviderOutcome",
    "classify_provider_exception",
    "TokenUsage",
    "extract_usage",
    "extract_cost",
]
