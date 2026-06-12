"""Token usage and cost read off a provider response: surfaced when reported,
None when absent — and cost is never computed from a pricing table."""

from drawbore.llm import TokenUsage, extract_cost, extract_usage


class _LiteLLMResponse(dict):
    """Mimics a litellm response: dict-accessible, with provider cost stashed on
    ``_hidden_params['response_cost']``."""

    def __init__(self, *, usage=None, response_cost=None):
        super().__init__()
        if usage is not None:
            self["usage"] = usage
        self._hidden_params = {} if response_cost is None else {"response_cost": response_cost}


def test_extract_usage_maps_provider_token_fields():
    resp = _LiteLLMResponse(usage={"prompt_tokens": 13, "completion_tokens": 28, "total_tokens": 41})
    assert extract_usage(resp) == TokenUsage(input_tokens=13, output_tokens=28, total_tokens=41)


def test_extract_usage_derives_total_when_provider_omits_it():
    resp = _LiteLLMResponse(usage={"prompt_tokens": 5, "completion_tokens": 7})
    assert extract_usage(resp) == TokenUsage(input_tokens=5, output_tokens=7, total_tokens=12)


def test_extract_usage_is_none_when_no_usage_reported():
    assert extract_usage(_LiteLLMResponse()) is None
    assert extract_usage(_LiteLLMResponse(usage={})) is None


def test_extract_cost_reads_provider_reported_cost():
    resp = _LiteLLMResponse(response_cost=0.0000755)
    assert extract_cost(resp) == 0.0000755


def test_extract_cost_is_none_when_provider_does_not_report_it():
    # No _hidden_params at all, and present-but-empty — both stay None (never
    # fabricated from a pricing table).
    assert extract_cost(_LiteLLMResponse()) is None
    assert extract_cost({"choices": []}) is None


def test_token_usage_to_dict_is_json_safe():
    assert TokenUsage(1, 2, 3).to_dict() == {
        "input_tokens": 1, "output_tokens": 2, "total_tokens": 3,
    }
