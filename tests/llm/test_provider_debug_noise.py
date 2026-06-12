"""The gateway owns its provider dependency's logging config.

The underlying provider SDK prints unsolicited debug blurbs (e.g. a "Provider
List" pointer) straight to stdout around failing model attempts, drowning a
program's own output. The gateway/adapter boundary must suppress that provider
debug printing centrally so a framework user never has to reach past the gateway
to configure the provider SDK directly. This asserts the verified provider flag
the adapters set; stdout-capture assertions would be brittle.
"""

import litellm

from drawbore.llm.gateway import LiteLLMGateway
from drawbore.llm.production import ProductionLLMGateway
from drawbore.llm import LLMRuntimeConfig


def test_litellm_gateway_suppresses_provider_debug_on_construction(monkeypatch):
    monkeypatch.setattr(litellm, "suppress_debug_info", False, raising=False)
    LiteLLMGateway()
    assert litellm.suppress_debug_info is True


def test_production_gateway_suppresses_provider_debug_on_construction(monkeypatch):
    monkeypatch.setattr(litellm, "suppress_debug_info", False, raising=False)
    ProductionLLMGateway(config=LLMRuntimeConfig())
    assert litellm.suppress_debug_info is True
