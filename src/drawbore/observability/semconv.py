"""GenAI Semantic-Convention attribute names + Drawbore span attributes.

The OTel GenAI Semantic Conventions live under
``opentelemetry.semconv._incubating.attributes.gen_ai_attributes`` — a PRIVATE,
unstable API path (the leading underscore is OTel's "may change without notice"
marker). The attribute *names* it defines, however, are spec-stable. Drawbore
therefore declares its own constants with those exact string values, so the
framework depends on the convention, not on a private module path.
"""

from __future__ import annotations

# --- GenAI Semantic Conventions (spec-stable names) ---
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_AGENT_VERSION = "gen_ai.agent.version"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"

# --- GenAI operation values (the span "operation" + the span-name prefix) ---
OP_EXECUTE_TOOL = "execute_tool"
OP_INVOKE_AGENT = "invoke_agent"
OP_CHAT = "chat"

# --- Drawbore-specific span attributes (outside the gen_ai namespace) ---
DRAWBORE_RUN_ID = "drawbore.run_id"
DRAWBORE_STEP = "drawbore.step"
DRAWBORE_STATUS = "drawbore.status"
DRAWBORE_TENANT_ID = "drawbore.tenant_id"
DRAWBORE_RISK_TIER = "drawbore.risk_tier"
DRAWBORE_INPUT_HASH = "drawbore.input_hash"
DRAWBORE_OUTPUT_HASH = "drawbore.output_hash"
DRAWBORE_TOOL_OPERATION = "drawbore.tool.operation"
DRAWBORE_DECLARED_MODEL = "drawbore.declared_model"
# How many bounded corrective structured-output reprompts a step's model turn took.
DRAWBORE_STRUCTURED_OUTPUT_REPROMPTS = "drawbore.structured_output.reprompts"

# --- Evidence compression ---
DRAWBORE_EVIDENCE_DECISION = "drawbore.evidence.decision"
DRAWBORE_EVIDENCE_TRANSFORM = "drawbore.evidence.transform"
DRAWBORE_EVIDENCE_HANDLE_ID = "drawbore.evidence.handle_id"
DRAWBORE_EVIDENCE_ORIGINAL_TOKENS = "drawbore.evidence.original_tokens"
DRAWBORE_EVIDENCE_COMPRESSED_TOKENS = "drawbore.evidence.compressed_tokens"
DRAWBORE_EVIDENCE_POLICY = "drawbore.evidence.policy"
