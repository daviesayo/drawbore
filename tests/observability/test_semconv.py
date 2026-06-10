from drawbore.observability import semconv


def test_genai_attribute_names_match_the_spec_strings():
    # Drawbore owns these constants (the opentelemetry.semconv._incubating
    # path is internal/unstable) but their values are the spec-stable names.
    assert semconv.GEN_AI_OPERATION_NAME == "gen_ai.operation.name"
    assert semconv.GEN_AI_AGENT_NAME == "gen_ai.agent.name"
    assert semconv.GEN_AI_AGENT_ID == "gen_ai.agent.id"
    assert semconv.GEN_AI_AGENT_VERSION == "gen_ai.agent.version"
    assert semconv.GEN_AI_TOOL_NAME == "gen_ai.tool.name"
    assert semconv.GEN_AI_REQUEST_MODEL == "gen_ai.request.model"
    assert semconv.GEN_AI_RESPONSE_MODEL == "gen_ai.response.model"


def test_operation_values_are_the_three_capture_points():
    assert semconv.OP_EXECUTE_TOOL == "execute_tool"
    assert semconv.OP_INVOKE_AGENT == "invoke_agent"
    assert semconv.OP_CHAT == "chat"


def test_drawbore_attributes_are_namespaced():
    assert semconv.DRAWBORE_RUN_ID == "drawbore.run_id"
    assert semconv.DRAWBORE_STEP == "drawbore.step"
    assert semconv.DRAWBORE_STATUS == "drawbore.status"
    assert semconv.DRAWBORE_TENANT_ID == "drawbore.tenant_id"
    assert semconv.DRAWBORE_RISK_TIER == "drawbore.risk_tier"
    assert semconv.DRAWBORE_INPUT_HASH == "drawbore.input_hash"
    assert semconv.DRAWBORE_OUTPUT_HASH == "drawbore.output_hash"
    assert semconv.DRAWBORE_TOOL_OPERATION == "drawbore.tool.operation"
    assert semconv.DRAWBORE_DECLARED_MODEL == "drawbore.declared_model"
