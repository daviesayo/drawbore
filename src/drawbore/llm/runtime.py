"""The single LLM runtime authority.

One ``LLMRuntime`` owns runtime config, profile/direct resolution, credential checks,
the per-attempt provider walk (fallback + classification), the ``ModelAudit``, and the
optional opaque ``model_factory`` slot the ADK loop reads. The ADK model factory
itself lives in ``drawbore.orchestration`` (containment); in production the slot is
None and the engine supplies its default; in test mode the harness sets it.
"""

from __future__ import annotations

from typing import Any, Callable

import litellm

from drawbore.errors import DrawboreError

from .attempts import ModelAttemptAudit, ModelAudit
from .build import build_model_request
from .classify import classify_provider_exception
from .config import LLMRuntimeConfig
from .credentials import CredentialChecker, EnvCredentialChecker, NullCredentialChecker
from .errors import LLMConfigError, LLMError, ModelUnavailableError
from .gateway import LLMGateway
from .production import ProductionLLMGateway
from .request import ModelResponse
from .resolution import ResolvedModelChain, resolve_chain


class LLMRuntime:
    """The single source of provider policy for one-shot AND loop paths."""

    def __init__(
        self,
        *,
        config: LLMRuntimeConfig | None = None,
        gateway: LLMGateway | None = None,
        credential_checker: CredentialChecker | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config if config is not None else LLMRuntimeConfig()
        self.gateway = gateway if gateway is not None else ProductionLLMGateway(config=self.config)
        self.credential_checker = credential_checker or EnvCredentialChecker()
        self.model_factory = model_factory

    @classmethod
    def from_gateway(
        cls, gateway: LLMGateway, *, model_factory: Callable[[str], Any] | None = None
    ) -> "LLMRuntime":
        """Compat path: an empty-config runtime wrapping ``gateway`` with a
        permissive credential checker. The empty config declares no profiles, so a
        spec must use direct model strings (a ``profile:`` ref would not resolve)."""
        return cls(
            config=LLMRuntimeConfig(), gateway=gateway,
            credential_checker=NullCredentialChecker(), model_factory=model_factory,
        )

    def resolve(self, spec) -> ResolvedModelChain:
        """Resolve the agent's declared model(s) to a concrete chain.

        For a ONE-SHOT agent (``not spec.tools``) whose resolved provider has
        ``native_structured_output=True``, fail closed at resolve time if the
        provider/model does not support native schema-constrained decoding — a
        misconfiguration caught before any model call, never a silent fallback.
        The ``not spec.tools`` gate is load-bearing: native output applies only to
        the one-shot path, so a model+tools loop agent is never blocked here.
        """
        chain = resolve_chain(spec, self.config, credential_checker=self.credential_checker)
        if not spec.tools:
            for attempt in chain.attempts:
                if attempt.provider is None:
                    continue  # providerless direct string → no provider config → no guard
                provider_cfg = self.config.providers.get(attempt.provider)
                if provider_cfg is None or not provider_cfg.native_structured_output:
                    continue
                if not litellm.supports_response_schema(
                    model=attempt.model, custom_llm_provider=attempt.provider
                ):
                    raise LLMConfigError(
                        f"native structured output enabled for model "
                        f"{attempt.model!r} on provider {attempt.provider!r}, which "
                        f"does not support it"
                    )
        return chain

    async def complete(self, spec, payload, chain: ResolvedModelChain) -> ModelResponse:
        """Walk ``chain`` calling the gateway once per attempt: fall back on a
        ``transport`` failure whose reason is in that attempt's ``fallback_on``; fail
        closed (no fallback) on auth (``model_config_error``) and contract
        (``model_error``); exhaustion -> ``model_unavailable``. Returns a
        ``ModelResponse`` carrying a full ``ModelAudit``."""
        audits: list[ModelAttemptAudit] = []
        for i, attempt in enumerate(chain.attempts):
            native = bool(
                attempt.provider
                and (pc := self.config.providers.get(attempt.provider))
                and pc.native_structured_output
            )
            request = build_model_request(
                spec, payload, model_chain=(attempt.request_model,), native=native
            )
            try:
                response = await self.gateway.complete(request)
            except LLMError:
                # Contract failure raised by the gateway (non-JSON / bad shape). No
                # no fallback — unless schema_violation is explicitly allowed,
                # which non-JSON is not. Record halted and re-raise (model_error).
                audits.append(self._attempt_audit(i, attempt, "halted", reason="contract"))
                raise
            except DrawboreError:
                # A Drawbore-internal fail-closed signal raised BY the gateway itself
                # (e.g. test mode's TestingError for a missing mock) is not a provider
                # transport failure — never classify it as one. Re-raise unchanged so
                # it carries its own legible halt_reason up the pipeline.
                audits.append(self._attempt_audit(i, attempt, "halted", reason="internal"))
                raise
            except Exception as exc:  # provider/transport — classify (never blind-fallback)
                outcome = classify_provider_exception(exc)
                if outcome.kind == "auth":
                    audits.append(self._attempt_audit(i, attempt, "halted", reason="auth_error"))
                    raise LLMConfigError(
                        f"provider {attempt.provider!r} rejected credentials for "
                        f"{attempt.request_model}: {exc}"
                    ) from exc
                if outcome.kind == "contract":
                    audits.append(self._attempt_audit(i, attempt, "halted", reason="contract"))
                    raise LLMError(
                        f"provider {attempt.provider!r} returned a bad request for "
                        f"{attempt.request_model}: {exc}"
                    ) from exc
                if outcome.kind == "transport" and outcome.reason in attempt.fallback_on:
                    audits.append(self._attempt_audit(i, attempt, "fallback", reason=outcome.reason))
                    continue
                # unknown, or a transport reason this profile does not fall back on:
                # fail closed without advancing (no broad fallback).
                audits.append(self._attempt_audit(i, attempt, "halted", reason=(outcome.reason or "unknown")))
                raise ModelUnavailableError(
                    f"provider attempt {attempt.request_model} failed and is not a "
                    f"configured fallback reason: {exc}"
                ) from exc
            else:
                audits.append(self._attempt_audit(i, attempt, "success"))
                return ModelResponse(
                    output=response.output, model_used=response.model_used,
                    raw_text=response.raw_text,
                    usage=response.usage, cost=response.cost,
                    reprompts=response.reprompts,
                    audit=ModelAudit(
                        declared_refs=chain.declared,
                        selected_provider=attempt.provider, selected_model=attempt.model,
                        attempts=tuple(audits),
                    ),
                )
        raise ModelUnavailableError(
            f"all model attempts failed for chain {[a.request_model for a in chain.attempts]}"
        )

    @staticmethod
    def _attempt_audit(index, attempt, outcome, *, reason=None) -> ModelAttemptAudit:
        return ModelAttemptAudit(
            index=index, declared_ref=attempt.declared_ref, source=attempt.source,
            provider=attempt.provider, model=attempt.model,
            request_model=attempt.request_model, outcome=outcome, reason=reason,
        )
