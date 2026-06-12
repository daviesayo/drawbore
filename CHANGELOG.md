# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-06-12

### Added
- `FileCheckpointStore` (from `drawbore.state`): a durable, file-backed
  checkpoint store that persists the full resume contract — step outputs, skip
  marks, the topology fingerprint, per-step seals, and trust labels — to a
  directory. A fresh instance over the same directory resumes a run across a
  process restart, restoring completed steps without re-execution and without a
  spurious `resume_drift`, while a genuine contract change still refuses. Each
  step is written atomically (temp file then rename), so a crash mid-write never
  tears a committed checkpoint. Outputs are stored as plain field data and
  reconstructed from the live pipeline's output schema, never from a class path
  read off disk, so a tampered checkpoint cannot trigger an arbitrary import.
  Stdlib only. Documented in the durable-resume guide.
- A non-JSON or empty model response on a one-shot model call is now re-attempted
  once on the same model before halting. This single, always-on, bounded retry
  clears the common transient case (an empty or truncated body) without a flag and
  without changing failure behaviour: a still-unusable response after the retry
  still halts with `model_error`, and a structural mismatch (wrong shape, or valid
  JSON that is not an object) still fails closed immediately with no retry. A
  model+tools tool-loop step is not retried, because re-running a loop could replay
  tool side effects. Documented in the production LLM gateway guide.

### Changed
- `CheckpointStore` gains an optional `bind_models` hook (default no-op). The
  pipeline supplies each step's live output model at run start so a serialising
  store can reconstruct typed outputs safely. Existing stores need no change.
- A `model_error` from a non-JSON or empty model response now includes a short,
  bounded excerpt of the offending output in its halt reason (and so in the audit
  record), so the failure is diagnosable without re-running. The excerpt is the
  model's own failed text, truncated to a fixed length; it is never an unbounded
  dump of model content. Documented in the production LLM gateway guide.

### Fixed
- Provider runtime config now reaches the agentic tool loop. A `ProviderConfig`'s
  `base_url`, `timeout_seconds`, and `extra` pass-through (for example
  `response_format={"type": "json_object"}` for provider-side JSON enforcement) were
  applied on the one-shot model path but dropped on the model+tools tool-loop path,
  whose model was built from the model name alone. Loop steps now resolve the same
  per-provider config as one-shot steps, so JSON-mode and transport settings behave
  identically on both paths. A caller-supplied model factory is still used as-is.
  Documented in the production LLM gateway guide.
- A model-backed agent that declares a tool whose ref is not a valid provider
  function name — an MCP tool (`mcp://server/tool`) or the evidence retrieval tool
  (`evidence://retrieve`) — no longer breaks the agentic tool loop on
  OpenAI-compatible providers. Function-calling providers restrict tool names to
  letters, digits, `_`, and `-`, so such refs could not be represented and the
  model's call failed to resolve. The loop now exposes each tool to the model under
  a deterministic, collision-free provider-safe alias and resolves the alias back to
  the canonical ref before the call runs. Scope enforcement, the one-time token, the
  proxy log, and the audit trail stay keyed on the canonical ref; an alias that maps
  to no declared tool is still refused. Documented in the agentic tool loop guide.

### Documentation
- The taint-and-trust guide now covers tools called from the agentic tool loop. A
  model-driven tool call reaches its handler through the same proxy as a deterministic
  agent's call, so the taint breaker is enforced identically — an exfil-capable call
  under an untrusted scope halts `taint_violation` whether the call came from your code
  or from the model mid-completion, and the provider-safe alias the model sees never
  weakens the breaker. The guide and the agentic tool loop guide also name the refusal
  shapes by halt code: in the loop, a call to a tool the agent did not declare (or an
  invented alias) never reaches the proxy and fails closed with `model_error`, whereas a
  deterministic agent calling a registered-but-undeclared tool directly halts
  `tool_access` — same fail-closed outcome, different halt code to alert on.

## [0.3.0] - 2026-06-12

### Added
- Safety gauntlet runners (`run_containment`, `assert_contained`, `run_pack`)
  now accept `llm_config=` and `credential_checker=` and forward them to
  `test_mode`, so the shipped containment corpus works on pipelines whose
  agents bind a model by profile (`model="profile:..."`). Without them a
  profile-bound pipeline halted `model_config_error` before the attack was
  provoked. As in all of test mode, the profile resolves against the credential
  checker and no provider is called.
- `register_mcp_server` now accepts optional per-tool `source_trust` and
  `exfil_capable` maps (keyed by tool name), so an outward-writing MCP tool — a
  notifier, an email relay — can be declared exfil-capable through the public MCP
  path and is then refused by the taint sink-gate exactly like a custom
  `register_tool(..., exfil_capable=True)`. A tool you do not name keeps the
  fail-safe default (untrusted-source, not a sink); a flag keyed to a tool not in
  `allowed_tools` raises `ValueError`. Documented in the MCP tools and taint-and-trust
  guides.
- `RunResult.metrics`: a quantitative cost and performance view of every run,
  separate from the legible audit trace. It carries per-step wall-clock
  `duration_seconds`, per-step model `tokens` (a provider-agnostic `TokenUsage` of
  `input_tokens` / `output_tokens` / `total_tokens`) and provider-reported `cost`,
  and the tool-proxy call log (`tool`, `operation`, `duration_seconds`, `result`).
  `tokens` and `cost` are `None` when the provider does not report them — cost is
  never fabricated from a pricing table. `metrics.to_dict()` renders the whole
  object to JSON-safe primitives.
- `TokenUsage` (from `drawbore.llm`) and `RunMetrics` / `StepMetric` /
  `ToolCallMetric` (from `drawbore.audit`) on the public surface.
- `ModelResponse` now carries `usage` and `cost` read off the provider response.
- Test mode: `pipeline.test_mode(mock_model_usage={...})` lets the fake provider
  report token usage for a one-shot model agent so `result.metrics` can be
  asserted in tests.

### Fixed
- The LLM gateway now silences the underlying provider SDK's unsolicited debug
  printing (e.g. the repeated "Provider List" pointer it writes to stdout around
  failing model attempts), so run output stays clean. Callers no longer need to
  import the provider SDK and set its debug flag themselves. This affects only
  the provider SDK's own debug prints; no safety, audit, or observability signal
  is suppressed.

## [0.2.0] - 2026-06-11

### Added
- Refuse-on-drift resume: resuming a checkpointed run now verifies a per-step
  seal of each completed step's declared contract (version, risk tier, tools,
  model, instructions, schemas, evidence policy) before anything executes, and
  halts with the new `resume_drift` halt code naming the changed field instead
  of replaying outputs into changed logic.
- `RunResult.resume_ledger`: a regulator-readable record of every step's
  resume disposition (`executed` / `restored` / `skipped` / `refused`) with a
  `legible()` rendering, present on every run.
- `CheckpointStore.record_seal` / `seal_of` with safe defaults; custom durable
  stores should implement both and persist output, trust, and seal atomically.
- New guide: durable resume (`docs/guide/durable-resume.mdx`).

### Changed
- Resuming under a changed pipeline topology now halts with `resume_drift`
  when the run had prior progress. Previously the stale checkpoints were
  silently discarded and the pipeline re-ran from the beginning under the same
  `run_id`; that silent restart could re-execute already-completed steps.
- Resumes from checkpoint stores that do not persist seals (including
  checkpoints written by earlier versions) now refuse with `resume_drift`:
  an unverifiable step is never restored. Re-run such flows under a fresh
  `run_id`, or upgrade the store.

## [0.1.0] - 2026-06-10

### Changed

- Development dependencies moved out of the public `dev` extra into PEP 735 dependency-groups.
- `pipeline.test_mode()` now exposes its full typed signature (was `**kwargs`) for IDE completion.
- Internal development annotations removed from `src/` and `pyproject.toml`.
  These identifiers were never intended for the public package surface;
  documentation and explanatory content is preserved, only the codenames are
  gone. A new automated test (`tests/docs/test_source_boundary.py`) asserts
  zero matches going forward.
- `import drawbore` no longer eagerly imports provider/engine SDKs for
  deterministic-only pipelines (~2.6 s → under 0.3 s); the cost moves to first
  engine construction.
- Schema-violation halt reasons use the stable token `schema_violation: input|output: ...`
  with a readable field-by-field summary (no raw error dumps); the audit record's
  violation count now keys on the token, not prose.

### Fixed

- `Pipeline.run()` raises a legible `TypeError` when the initial input is not a
  Pydantic model (was a raw `AttributeError`).
- Checkpointed runs fail closed: `Pipeline.run(checkpoints=...)` without an explicit
  `run_id` now raises instead of silently replaying a previous run's outputs.
  `RunResult` gains `run_id` so callers can see which run identity executed.
- Resolver fail-closed completeness: `from_json` now surfaces malformed JSON as `ConfigResolutionError` (was a raw `json.JSONDecodeError`), so the resolver's contract — it raises only `ConfigResolutionError` — holds for parse errors too.
- In-loop tool denial no longer prints an engine-internal traceback or an OpenTelemetry context-detach error to stderr. The success/failure output of examples like the taint-gated demo is now clean; exit code and runtime behaviour are unchanged.

### Added

- PEP 561 `py.typed` marker: type checkers now see Drawbore's annotations.
- `JoinNode` is exported from `drawbore.pipeline` (it is `Join`'s return type).
- `RunResult.halt_code`: a stable, documented halt token (`drawbore.errors.HALT_CODES`)
  so callers branch on a code instead of parsing the human-readable `reason`.
- Taint-gated tool calls: declare `source_trust` / `exfil_capable` on tools and Drawbore
  refuses to invoke an exfil-capable tool while a step is handling untrusted data — a
  deterministic lethal-trifecta breaker (untrusted input, tool access, outward data flow). Untrusted data also cannot steer a `When` branch,
  and trust survives checkpoint resume. MCP tools are untrusted-source by default; set the
  pipeline's `initial_trust` for untrusted-ingest (triage) workflows.
- Safety gauntlet: a reusable test-mode corpus that drives canonical attacks (schema
  violation, low-confidence escalation, unmocked-tool fail-closed, circuit-breaker trip,
  parallel-call refusal, non-JSON model output) through the real safety layer and asserts
  structural refusal — `drawbore.testing.run_containment` / `assert_contained` / `run_pack`
  with `Containment` verdicts. `None` means the attack was not contained.
- Authority regression check: derive a capability footprint from a pipeline manifest and
  fail closed when an edit grants an agent new tool or evidence authority
  (`drawbore.config.check_no_new_authority`, `authority_diff`, `effective_authority`),
  with a `test_mode` soundness helper (`drawbore.testing.assert_footprint_sound`).
- Native branching with serializable `When` conditions, explicit `Join` nodes
  (`exactly_one` / `first_by_priority` / `all_present`), and skip-propagation —
  every not-taken branch is explained in the audit trace. Pipeline JSON config
  round-trips the new constructs.
- Runnable examples: `examples/tiny_pipeline/` and `examples/remittance_validation/`, each with a README and tests that run via `pipeline.test_mode` (synthetic data, no live calls), plus example pages under the docs Examples section. Examples now run in CI.
- Public documentation site skeleton: a `docs.json` navigation, a docs landing page and quickstart, coding-agent docs (`docs/agents/*`), and the guides reorganized into a clear public structure.
- MIT `LICENSE` file at the repository root (the package already declared MIT).
- Engine carrier: `StepExecution(output, model_audit, model_turns)`; `ADKEngine` consumes the one `LLMRuntime` (one-shot resolves and completes through it), with `gateway=` kept as a back-compat alias.
- Runtime LLM authority: `LLMRuntime` resolves provider chains, walks fallbacks with classification, and records a `ModelAudit`; `from_gateway(...)` keeps direct model strings working.
- Production gateway: single-call `ProductionLLMGateway` (per-provider base_url/timeout/extra; contract failures fail closed; transport exceptions propagate for the runtime to classify); `ModelResponse.audit` additive field.
- Provider classification: `classify_provider_exception` maps LiteLLM exceptions to the closed `transport`/`auth`/`contract`/`unknown` taxonomy — no broad catch-and-fallback.
- Model audit: `ModelAttemptAudit`/`ModelAudit` (declared refs, attempts, selected model, loop fallback phase) with a legible summary.
- Audit: `StepAuditRecord.model` carries the per-step `ModelAudit`; the pipeline records the model attempts (rendered in `audit_trace.legible()`) and reads `model_turns` from the `StepExecution` carrier.
- Model resolver: `resolve_chain(...)` -> `ResolvedModelChain` of `ModelAttempt`s; profile expansion, direct-string passthrough, `(provider, model)` dedup, credential preflight.
- Runtime LLM config: typed, closed `LLMRuntimeConfig`/`ModelProfile`/`ModelTarget`/`ProviderConfig` (separate from pipeline JSON).
- Failure taxonomy: `LLMConfigError` (halt reason `model_config_error`) and the `profile:` grammar.
- Credential seam: `CredentialChecker` protocol + `EnvCredentialChecker` (availability only, never the secret value).
- Loop fallback: model+tools agents resolve through the runtime; provider fallback before any tool call may advance, a post-tool provider failure fails closed, and tool failures/denials always abort — bounded by `max_llm_calls`.
- Test mode: `profile:*` resolves under test mode via a per-step `LLMRuntime`; `StaticCredentialChecker` (default available) keeps CI key-free, while `available=False` exercises `model_config_error`; missing mocks still fail closed with `TestingError` — never a provider call.
- Observability + docs: chat spans tag the declared ref (`drawbore.declared_model`) and resolved model; containment tests keep LiteLLM under `drawbore.llm` and ADK under `drawbore.orchestration`; the production-LLM-gateway guide.
- Acceptance tests: end-to-end provider resolution, one-shot fallback audit, missing-credential `model_config_error`, and `LocalEngine` engine-agnostic fail-closed through `pipeline.run`.
- Phase 1 release readiness for the `v0.1.0` tag.
- Validation-pipeline docs: the canonical validation-pipeline example (`docs/examples/remittance-validation.mdx`).
- Acceptance (config composition): tests proving the canonical pipeline exports to JSON config with named `AgentCatalog` refs, loads back through `from_json(data, agents=catalog, registry=registry)` against a pre-populated registry, runs config-loaded in test mode with mocks supplied outside the config, rejects version drift before test mode, and never serializes mocks/stores/seeded handles/model scripts.
- Acceptance (tool access): a test proving a declared tool with no mock fails closed in test mode (`testing_error`), halts at `transaction_retriever`, and records a failed step where the tool call was attempted — no live external is reached.
- Acceptance (schema violation): a test proving a mocked model response that violates a downstream output schema halts at `risk_scorer`, records one schema violation, and runs no step past the failed boundary.
- Acceptance (escalation): a negative case proving a low-confidence `risk_scorer` output (a `HasConfidence` model below the pipeline threshold) escalates through the configured policy with a legible reason on the run and audit record; the canonical happy path stays zero-escalation.
- Acceptance (authority-bound retrieval): tests proving the `evidence_reviewer` loop reaches `evidence://retrieve` via proxy/JIT with an explicit seeded handle — allowed search succeeds and is audited, and a policy-forbidden full retrieval fails closed and halts the step.
- Acceptance (evidence compression): a test proving the `risk_scorer` step compresses large model-bound evidence, records a compressed decision and handle in audit, and retains the original in the injected evidence store — distinct from the seeded review handle.
- Acceptance (audit legibility): a test asserting `audit_trace.legible()` carries the pipeline name/version, completed-step count, each tool call, model turns, the evidence-compression decision and handle, and the final status — durable phrases, not a snapshot.
- Acceptance (happy path): the canonical five-step remittance_confirmation pipeline (deterministic + tool-backed + model one-shot + model+tools loop + fan-in) runs end-to-end through `pipeline.test_mode` with explicit `From(...)` bindings and completes with a typed confirmation, five audited steps, and zero escalations/violations.
- Validation scenario: locked refs/models/tools for the canonical remittance-confirmation acceptance pipeline (`tests/acceptance/test_remittance_validation.py`).
- Test-mode hardening: `pipeline.test_mode(...)` now validates `mock_model_responses` and `mock_loop_scripts` keys eagerly at construction — each must name a step of the matching category (a one-shot model agent for responses, a model+tools agent for loop scripts), else a legible `TestingError`. Previously a mistyped or miscategorized agent-name key did nothing until a run-time fail-closed halt; this mirrors the existing eager tool-mock guard.
- Test-mode containment: a test enforcing that `drawbore.testing` imports no `google.adk`/`litellm`/MCP SDK — the fake loop model (a `BaseLlm`) lives under `drawbore.orchestration` and is reached only as an opaque model factory.
- Test-mode acceptance (config + safety): tests proving a `from_json` pipeline runs in test mode without weakening config drift checks, mocks never leak to the global registry, declared externals fail closed by default, and `allow_real_tools` is the only real-tool opt-in.
- Test-mode acceptance (evidence): tests proving evidence policy disabled/simulate/compress behave as designed in test mode, and `evidence://retrieve` is auto-bound to the test store, runs through the proxy for both allow and deny, and shadows any original-store binding.
- Test-mode acceptance (loop): tests proving the scripted loop drives the real agentic tool loop — happy path (tool call + final JSON, model turns audited), in-loop tool denial (immediate abort + failed-step audit), multi-call fail-closed, the loop bound, and agent-name routing for two same-`model` loop agents.
- Test-mode acceptance (model): tests proving mocked model outputs flow through the real escalation/confidence and human-approval gates, agent-name routing distinguishes two same-`model` agents, a missing response fails closed, and a fallback chain is exercised without a live provider.
- Test-mode acceptance (failures): tests proving local test mode preserves the real input/output schema gates, proxy-routed mocked tool calls and step audit, operation-scope denial, and the per-step circuit breaker.
- Test-mode harness: `Pipeline.test_mode(...)` returns an async context manager (`drawbore.testing.TestMode` -> `TestPipeline`) that runs the pipeline through the REAL `Pipeline.run` with a scoped registry overlay, a name-routing `TestEngine`, an in-memory audit sink and evidence store, and deterministic run ids. `TestPipeline.run` returns a production-shaped `RunResult`/`audit_trace`; `.audit_sink`/`.evidence_store` are exposed. No global state is mutated.
- Test engine: `drawbore.testing.engine.TestEngine` — an `OrchestratorEngine` that routes by `AgentSpec.name` and delegates each step to a per-step `ADKEngine` (agent-scoped `FakeGateway` for one-shot, agent-scoped scripted `model_factory` for the real agentic loop; deterministic agents run their real `fn`); missing model/loop mocks fail closed.
- Loop-script vocabulary: `drawbore.testing.{call,final,text,multi_call}` describe scripted model turns for a model+tools agent; `to_turns` normalizes them to the orchestration scripted-model tuple form (`final` becomes a JSON text turn).
- Scripted loop model: `drawbore.orchestration.make_scripted_model_factory` builds an ADK-contained fake `BaseLlm` (scripted call/multicall/text turns) handed to `ADKEngine(model_factory=...)` so local test mode drives the real agentic loop with no network; `drawbore.testing` reaches it only as an opaque callable.
- Fake gateway: `drawbore.testing.gateway.FakeGateway` — a pure `LLMGateway` (no ADK) that returns scripted one-shot `ModelResponse`s for a single named agent (dict/model/sequence/callable(ModelRequest)); missing/exhausted fails closed with `TestingError`; never infers the agent from prompt or model name.
- Scoped registry: `drawbore.testing.tools.build_scoped_registry` builds a per-test `ToolRegistry` overlay of only the declared tools, preserving each tool's allowed-operations/schema/kind; mock handlers still run through `ToolProxy`; unmocked declared tools fail closed at invoke; `allow_real_tools` opts a tool into its real handler; `evidence://retrieve` rebinds to the test evidence store. Mock value forms: static / sync / async / sequence (`drawbore.testing.models`).
- Runtime seam: an internal `registry_override` keyword on `Pipeline.run`/`_run_inner` (default `None`, backward-compatible) routes a single scoped `ToolRegistry` to BOTH the `ToolProxy` and the `ToolLoopBundle` for a run — the seam local test mode uses; the public mock surface remains `Pipeline.test_mode`.
- Test-mode foundation: `drawbore.testing.TestingError` (a `DrawboreError` with a self-declared `halt_reason="testing_error"`) and decisions for local test mode.
- Test-mode docs: a local-testing guide (`docs/guide/local-testing.mdx`).
- JSON-config docs: a JSON-config guide (`docs/guide/json-config.mdx`).
- Config containment: a test enforcing that `drawbore.config` imports no `google.adk`, `litellm`, MCP SDK, or `drawbore.orchestration`/`llm`/`mcp`/`audit`/`observability` — the config layer is an ADK-free construction layer over existing declarations that computes its own `schema_fingerprint` rather than reaching for the observability/exporter surface.
- Config round-trip coverage: linear and fan-in (`From("agent.field")`) pipelines, escalation policy + confidence threshold, per-step `EvidencePolicy` (applied to `Step.evidence`), one-shot model metadata, and model+tools-without-fallback all round-trip through `to_config`/`from_config` without importing ADK or constructing an engine; `to_json` is deterministic.
- Config import: `from_config(config, *, agents, registry=None)` / `from_json(data, *, agents, registry=None)` resolve a manifest into a live `Pipeline` — fail-closed on unknown `schema_version`, missing/ambiguous refs, declaration drift (version/risk-tier/tools/model/fallback/instructions/schema), duplicate names, `model+tools+fallback_model` and literal bindings, invalid tagged input modes / lying `depends_on`, and unregistered tools; static compatibility still runs through `Pipeline.add` and surfaces as `ConfigResolutionError`.
- Config export: `to_config(pipeline, *, agents)` / `to_json(...)` walk a live `Pipeline` to a typed manifest — agent declarations (ref via the bidirectional catalog, schema evidence + full SHA-256 hashes), tagged input modes derived from runtime semantics, serialized per-step `EvidencePolicy`. Export fails closed on an uncataloged/ambiguous agent and on any live step whose `depends_on` lies about the derived input mode. `to_json` is deterministic (sorted canonical JSON).
- Agent catalog: `drawbore.config.AgentCatalog` (bidirectional `ref <-> Agent`) plus `resolve_ref`/`ref_for` helpers that also accept a plain `Mapping[str, Agent]`, inverting it by object identity and failing closed on uncataloged (zero) or ambiguous (multiple) refs.
- Config models: the `extra="forbid"` manifest models in `drawbore.config.models` (`PipelineConfig`, `PipelineMetaConfig`, `OnFailureConfig`, `AgentConfig`, `StepConfig`, `StepInputConfig`, `BindingConfig`, `EvidencePolicyConfig`) — tagged input modes (`initial`/`previous`/`bindings`), literal bindings rejected with a legible message, unknown fields rejected at every level.
- Config foundation: `drawbore.config.ConfigResolutionError` (a `DrawboreError` with a self-declared `halt_reason="config_resolution_error"`) and `config.fingerprint.schema_fingerprint` (canonical-JSON + full SHA-256 schema-drift fingerprint).
- Agentic-tool-loop docs: an agentic-tool-loop guide (`docs/guide/agentic-tool-loop.mdx`); a pipeline test proving the agentic tool loop composes with evidence compression (a compressed model+tools step retrieves the original mid-loop via the proxy-scoped `evidence://retrieve` tool).
- Agentic pipeline wiring: for a model-backed step that declares tools the pipeline builds a `ToolLoopBundle` (proxy/issuer/registry/declared/run-ctx) and passes it to `engine.run_step(..., tool_loop=)`, so the engine drives the proxy-scoped loop; every step records `StepAuditRecord.model_turns` (0/1/N), and a step that halts after running tools (the loop's denied-tool case) is recorded as a failed step with its tool calls and reason. Evidence compression still runs before the loop and `evidence://retrieve` is callable in it.
- Agentic engine routing: `ADKEngine` routes a model-backed agent that declares tools through the hidden ADK Runner loop; one-shot model agents keep the gateway path and deterministic agents the in-process path (unchanged). `ADKEngine(gateway, *, model_factory=…, max_llm_calls=8)` — `model_factory` is duck-typed (default builds ADK's `LiteLlm`; the test seam), `max_llm_calls` is validated `>= 1`. A loop agent that also sets `fallback_model` fails closed with a legible reason.
- Agentic loop failure semantics: any in-loop tool failure aborts immediately (the engine hard-breaks the event stream and the loop re-raises the precise tool error — no further model turn or tool call runs); a runaway loop is bounded by `max_llm_calls` and halts as `LLMError`; a non-JSON / non-object final answer fails closed; and more than one function call in a single turn fails closed (one-call-per-turn).
- Agentic loop: `run_agentic_loop(spec, payload, *, tool_loop, run_id, model_factory, max_llm_calls)` drives a hidden ADK `Runner`/`LlmAgent` for a model-backed agent that declares tools — proxy-backed schema tools, the prompt from `build_model_request`, each tool call through the proxy + JIT, model turns traced and counted. Returns `(final_json_dict, model_turns)`; the pipeline validates the dict. Tested with a fake ADK `BaseLlm` (no network).
- Agentic loop callbacks: `make_loop_model_callbacks` traces each in-loop model turn as a `chat` span nested under `invoke_agent` and counts turns, and short-circuits (returns a terminating response) once a tool failure is recorded; `make_loop_before_tool_callback` blocks undeclared tools and blocks any tool after a failure — the secondary guard behind the engine's hard-break.
- Proxy-backed ADK tool: `proxy_backed_tool(ref, proxy=, issuer=, run_ctx=, schema=, failures=)` wraps a declared Drawbore tool as a custom ADK `BaseTool` whose declaration is built from the registry tool's JSON schema (so the model sees its parameters) and whose execution routes through `proxy.invoke` + a single-use JIT token; any failure is recorded in the bundle's `failures` and re-raised (immediate-abort signal). ADK stays under `drawbore.orchestration`.
- Agentic audit: `StepAuditRecord.model_turns` (0 deterministic / 1 one-shot model / N loop) makes every model turn auditable, not only traced; `AuditRecorder.record_failed_step` records a step that halted after doing work (e.g. a denied in-loop tool call) with its tool calls and reason, present in the trail but not counted in `steps` (success-only) — so the loop's headline failure mode is legible in the audit, not just the spans.
- Engine seam: `OrchestratorEngine.run_step` gains an optional `tool_loop` (a `ToolLoopBundle` carrying the proxy/issuer/registry/declared-refs/run-ctx plus mutable `failures`/`turns` accumulators) for the in-step tool loop; `LocalEngine` now fails closed with an engine-agnostic `EngineError` when asked to run a model-backed agent (it has no model path) instead of silently calling `spec.fn`.
- Evidence pipeline wiring: `Pipeline.add(agent, ..., evidence=EvidencePolicy(...))` (per-step) and `Pipeline.run(..., evidence_store=...)`. For a model-backed step with an enabled policy, the pipeline compresses the model view before the engine runs, re-validates the bounded view against the input model (never sends an invalid view — strict halts, else passthrough), stores the original, records the decision on the step's audit record (`StepAuditRecord.evidence`, rendered in `legible()`), and tags the `invoke_agent` span with `drawbore.evidence.*`. A fail-closed evidence failure halts-and-escalates with `evidence_error`. Deterministic and policy-less steps are unchanged.
- Evidence retrieval tool: `register_evidence_tool(registry, store=...)` registers a declared, proxy-backed builtin tool (`evidence://retrieve`) that serves full/search retrieval from the store, gated by per-handle policy (default search-allowed/full-denied) and expiry, failing closed. Search is bounded (capped result count) so it stays scoped and cannot reconstruct the full original. Reached only through the proxy + single-use JIT token — a handle visible in context does not by itself authorize retrieval (the proxy is authoritative). The registry is duck-typed so `evidence` imports nothing from `tools`.
- Evidence compression entrypoint: `compress_for_model(payload, policy, *, store, run_id, step, source_agent)` — routes to a transform, honours the mode (disabled/simulate/compress), gates on `min_tokens`, stores the original with a deterministic handle, and returns `(model_view, decision, handle)`. Disabled = byte-identical passthrough; simulate records the decision without changing the payload; compress fails closed (`EvidenceStoreError`) if the original can't be stored and is required. Pure (no pipeline/audit/observability import).
- Evidence logs transform: a deterministic `EvidenceTransform` that compresses long log text to errors, warnings, stack traces, and head/tail context with a dropped-line summary, keying on line-level severity markers (so a benign mid-sentence "error" is not a false ERROR line). Same input + policy gives byte-identical output.
- Evidence json_rows transform: a deterministic `EvidenceTransform` (protocol + name→transform registry) that compresses large record lists to a bounded, order-preserving view — head + tail + structurally-notable rows (flagged by a signal field, not benign prose) + an evenly-spaced sample, optionally capped to an output-token budget, with warnings. Same input + policy gives byte-identical output. `EvidencePolicy` gains `max_output_tokens`.
- Initial Phase 1 SDK scaffold: package layout for all modules, the `OrchestratorEngine` abstraction, and project metadata.
- Foundation: strict runtime schema validation, structural static compatibility, the `@agent` decorator + `AgentSpec`, `RunState`, the `OrchestratorEngine` ABC with an in-process `LocalEngine`, and `Pipeline` with `From` data bindings and a run loop that validates every boundary and halts on violation. Deterministic fan-in pipelines run end-to-end with no ADK.
- Tool access layer: `tools` package with a tool registry (custom + built-in), single-use opaque non-serializable capability tokens, an in-path tool proxy that enforces tool-level scope, logs calls, and trips a per-run circuit breaker, and a `ToolContext` that exposes only an agent's declared tools (run context flows via a contextvar, never through the agent). Agents declare tools via `@agent(tools=[...])`; `Pipeline(registry=...)` validates declared tools at registration. Still no ADK.
- Context isolation and errors/checkpointing: a `context` module (`build_input` for orchestrator-constructed isolated payloads; `sanitize` for structural bounds on external input), an `errors` module (`DrawboreError` + `halt_reason_for` taxonomy), and a `CheckpointStore` ABC with an in-memory default. The pipeline now halts-and-escalates on every failure — tool-misuse and agent errors return a halted `RunResult` instead of propagating — and a halted run resumes from the last successful step when given a stable `run_id` and a checkpoint store.
- Human escalation primitives: an `escalation` module with `EscalationPackage` (carrying a legible, non-engineer-readable summary via `legible()`), `EscalationPolicy` (sync/async delivery modes), an `EscalationDispatcher` interface plus an in-process `RecordingDispatcher` default, and the opt-in `HasConfidence` marker (only models that explicitly inherit it are confidence-checked). Agents may declare `@agent(requires_human_approval=True)`, recorded on `AgentSpec` (default `False`). The pipeline now turns any halt into a dispatched escalation when an `on_failure` policy is configured (`RunResult.status="escalated"`, `RunResult.escalations`), escalates on a `HasConfidence` output below the declared `confidence_threshold` (synchronously, or async = dispatch-and-continue), and treats `@agent(requires_human_approval=True)` as a synchronous approval gate. Without a policy, failures still halt as before (`status="halted"`).
- Identity and versioning (agent fields): `AgentSpec` and `@agent` now carry a `risk_tier` (`low`/`medium`/`high`/`critical`, default `low`) and a `version` (default `0.0.0`) — the inputs the identity and versioning layers consume.
- Identity value object: an `identity` module with `AgentIdentity` (the immutable id/ttl/purpose/risk_tier/sponsor/delegation/created-at record) and `attestation_surface` (the tool + input/output schema + risk-tier fingerprint whose change forces re-attestation).
- Identity registry: an in-process `IdentityRegistry` enforcing the identity lifecycle invariants — registration requires a human sponsor (no agent without an owner), a strict `draft → active → suspended → decommissioned` lifecycle, single-action atomic decommission, and a re-attestation block (`run_block_reason`) when an agent's attestation surface (tools/input/output/risk-tier) changes until the human sponsor re-attests.
- Versioning classifier: a `versioning` module with `classify_change(old_input, old_output, new_input, new_output)` that labels a change `breaking` (removed field, changed type, modified required-ness, or an added required field) or `non_breaking` (added optional field, or schema-invisible prompt/model swaps), via Pydantic field introspection.
- Versioning deployment: `RolloutPlan` + `Deployment` — the in-core staging state machine for the "never big-bang deploy" rule (shadow→canary→full for breaking changes, canary→full for non-breaking; a stage must pass before advancing; instant rollback to the last stable version is always available and a failed gate routes to rollback, not forward). Live traffic-splitting and the rollback dashboard are managed-service.
- Identity gate (pipeline): `Pipeline.run(..., identities=IdentityRegistry)` blocks a registered agent that is suspended, decommissioned, or pending re-attestation, halting-and-escalating through the escalation path (`reason="identity_<state>"`); an unregistered (draft) agent runs exactly as before. The `EscalationPackage` now carries the `agent_id` (the slot escalation reserved), populated on identity-gated halts and shown in `legible()`.
- Agent model fields: `AgentSpec`/`@agent` now carry `model`, `fallback_model`, and free-text `instructions` (all default `None`). An agent with `model=None` is a deterministic agent and is unchanged; an agent with a `model` is backed by the LLM gateway via the ADK engine.
- LLM boundary (core): a `llm` module with explicit `ModelRequest`/`ModelResponse` (Drawbore-owned, not ADK/LiteLLM internals), `resolve_model_chain` (primary→fallback), and `build_model_request` — the single, centralised model-input assembly point and the documented seam for the future pre-model evidence transform. No ADK.
- LLM gateway: `LLMGateway` (ABC) + `LiteLLMGateway` — non-streaming completion with a request-time fallback chain via `litellm.acompletion`. The gateway parses the model's JSON output into a `dict` (the pipeline validates it), falls back model-by-model on provider failure, and fails closed (`ModelUnavailableError`) when the whole chain fails or (`LLMError`) when a model returns non-JSON. Bifrost can later implement the same ABC.
- ADK tool guardrail: `drawbore.orchestration.adk_tools` exposes a declared Drawbore tool to ADK as a `FunctionTool` whose callable is the proxy-backed invocation (authoritative) plus a `before_tool_callback` that hard-blocks any undeclared tool (the second, independent guardrail). `google.adk` is imported only under `orchestration` — enforced by a test.
- ADK engine: `drawbore.orchestration.ADKEngine` (an `OrchestratorEngine`). Deterministic agents (no `model`) run in-process identically to `LocalEngine`; model-backed agents have their completion performed by the injected `LLMGateway` with the prompt built by `build_model_request`, returning the raw structured output for the pipeline to validate. A `ModelUnavailableError` from the gateway propagates for the pipeline to halt-and-escalate.
- Model-backed pipelines: a model-backed agent runs end-to-end through `Pipeline.run(engine=ADKEngine(gateway=...))` with every Drawbore guarantee intact — the model's structured output is validated against the agent's output model at the boundary, a schema violation halts, and a model-unavailable failure halts-and-escalates through the escalation path. Deterministic and model-backed agents compose in one pipeline.
- MCP errors and extra: an `mcp` module with `MCPError`/`MCPAuthError`/`MCPToolNotFoundError`; `MCPError` self-declares `halt_reason="mcp_error"` so a failed MCP tool call escalates legibly. The MCP SDK ships opt-in via the `drawbore[mcp]` extra (`google-adk[mcp]`) — the core dependency set is unchanged.
- MCP transport seam: `MCPClient` (ABC — connect/list_tools/call_tool/close), `MCPServerConfig`, `MCPToolSpec`, `OAuthConfig` (server-level OAuth 2.1/PKCE), and a dependency-free `FakeMCPClient` test double. The seam keeps the MCP security model unit-testable without a live server or the `mcp` SDK.
- MCP registration: `drawbore.mcp.register_mcp_server(registry, name=, url=, allowed_tools=, auth=, client=)` connects (server-level OAuth), validates each declared tool exists, and registers ONLY the declared tools as proxy-backed `Tool`s (`kind="mcp"`) under `mcp://<server>/<tool>` — registering a server does not expose its other tools. A declared tool the server doesn't advertise raises `MCPToolNotFoundError`. The registry gains a generic, MCP-agnostic `register_mcp_tool` (no `mcp` import).
- MCP security separation (proven): an MCP tool is reached only through the existing in-path proxy with a single-use JIT token scoped to that exact `mcp://<server>/<tool>` ref; an agent cannot reach an undeclared tool on the same server (blocked at both registration scope and proxy scope); server-level OAuth is consumed only at registration and never flows through the invocation path; the per-step circuit breaker applies to MCP tools. No new proxy authority was added.
- MCP-backed pipelines: a deterministic agent declaring `@agent(tools=["mcp://<server>/<tool>"])` calls the MCP tool through the existing `ToolContext`/proxy path and runs end-to-end through `Pipeline.run(...)` with every Drawbore guarantee intact — an MCP-tool failure halts-and-escalates with the legible `mcp_error` reason, and declaring an unregistered MCP tool is rejected at pipeline registration. `pipeline.py` is unchanged; the engine/proxy abstractions already carry MCP tools.
- Live MCP transport: `drawbore.mcp.live.LiveMCPClient` talks to real MCP servers over Streamable HTTP using the `mcp` SDK (the `drawbore[mcp]` extra), with server-level OAuth. It is import-guarded — using it without the extra raises a legible `MCPError` (`pip install drawbore[mcp]`) — and is the only `mcp`-SDK user, enforced by a containment test. A real server round-trip is not unit-tested (no live server in CI); the security model is proven against `FakeMCPClient`.
- Observability core: a leaf `observability` module — OTel GenAI-semconv span helper (`genai_span`), the Drawbore-owned GenAI attribute-name constants (`semconv`, so the framework depends on the spec, not the private `opentelemetry.semconv._incubating` path), a shared `payload_hash` identity scheme, a test-isolated tracer-provider override (`use_tracer_provider`/`reset_tracer_provider`), and `ObservabilityError`. No new dependency — `opentelemetry-api`/`-sdk` are already core.
- Tool-call spans: `ToolProxy.invoke` now emits an `execute_tool {tool}` OTel span tagged with `gen_ai.tool.name`, the run id, step, operation, and input/output hashes, with ERROR status and the denial label (`denied:token`/`denied:scope`/`denied:breaker`) on a blocked call. The proxy's input/output hashing now routes through the shared `payload_hash`; `self.log` is unchanged (the span is additive).
- Agent-invocation spans: the pipeline emits an `invoke_agent {name}` OTel span around each step, tagged with `gen_ai.agent.name`/`.version`/`.id` (when identity-gated), risk tier, run id, step, tenant id (new optional `Pipeline.run(..., tenant_id=...)`), and the input hash; a failed step marks the span ERROR. The step's `execute_tool` and `chat` spans nest under it; spans are correlated across a run by the `drawbore.run_id` attribute.
- Model-call spans: the ADK engine's model path emits a `chat {model}` OTel span around the gateway completion, tagged with `gen_ai.request.model` (the resolved primary) and `gen_ai.response.model` (the model actually used after any fallback). It nests under the step's `invoke_agent` span; deterministic agents emit no `chat` span.
- Audit primitives: an `audit` module — an immutable, regulator-legible `AuditRecord` (with `StepAuditRecord` per-step detail and a `legible()` rendering), an append-only `AuditSink` ABC + in-process `InMemoryAuditSink` default (queryable by run id), and an `AuditRecorder` that builds the record during a run. The core is the basic readable/exportable log; tamper-evidence/crypto-signing/compliance export are managed-service.
- Audit wiring: every `Pipeline.run(...)` now produces an `AuditRecord` on `RunResult.audit_trace` (always built — a completed run reports `audit_trace.steps`/`.escalations`/`.schema_violations`) and writes it to an injected `AuditSink` via the new `run(..., audit=...)` parameter. Each successful step's record carries the agent/version/id, input/output hashes, and a summary of the tool calls it made; a halt records the stopping reason. Finalization happens at a single point, so all the existing halt-and-escalate exits are unchanged.
- OTLP export: `drawbore.observability.configure_otlp_export(endpoint, *, headers=, set_global=)` wires a `TracerProvider` that batches Drawbore's spans to any OTLP drain (Axiom/Datadog/Grafana/Honeycomb). The exporter ships in the optional `drawbore[otlp]` extra — calling it without the extra fails closed with a legible `ObservabilityError` (`pip install drawbore[otlp]`). Spans always emit to the configured provider, so the core needs only `opentelemetry-api`/`-sdk`.
- Observability containment and docs: a containment test locks the layering invariant — `observability`/`audit` import no `google.adk` and no `mcp` SDK, and `observability` stays a leaf (no `agent`/`pipeline`/`tools`/`orchestration`/`llm`/`mcp` imports). Added the observability/audit user guide (`docs/guide/observability-audit.mdx`).
- Evidence primitives: a new `evidence` extension module — `EvidencePolicy` (opt-in, per-step; default disabled = passthrough), immutable `EvidenceHandle`/`EvidenceDecision` records (with `legible()`), a dependency-free deterministic `estimate_tokens` (char/4 heuristic — no tokenizer dependency), and `EvidenceError`/subclasses that self-declare `halt_reason="evidence_error"` so a fail-closed evidence failure escalates legibly.
- Evidence store: `EvidenceStore` (ABC) + `InMemoryEvidenceStore` — retains the original alongside the compressed view keyed by a deterministic handle id, serves metadata (no content leak), full retrieval, and bounded deterministic search within the original, and fails closed on an unknown or expired handle (injectable clock for expiry). Durable stores are managed-service.

### Changed

- Config amendment: `from_config`/`from_json` no longer reject a `model + tools + fallback_model` manifest; execution enforces the safe loop fallback rules. Import stays string-preserving and runtime-free.
- Optional dependencies: added the `otlp` extra to `pyproject.toml` for OTLP span export; core dependencies remain exactly the four runtime libraries.
- Runtime hardening: `Pipeline.add(...)` now rejects a second step that reuses an existing agent name (`ValueError`) instead of silently overwriting `_by_name`/`outputs` state — the runtime keys steps and outputs by agent name, so names must be unique within a pipeline.

### Fixed

- Tool-access layer security hardening: closed a proxy bypass where an agent could reach the raw tool handler and self-mint tokens for undeclared tools — agents now receive only proxy-backed shims with no reference to the proxy/issuer/registry. The circuit breaker is now per-agent (not per-run) and counts only authorised calls; the proxy logs every call including denials — token/scope/breaker — and handler errors; capability tokens enforce operation scope on consume; `register_tool` accepts a `schema` and tools record `kind` provenance; `get_run_context` raises a typed `ToolAccessError`.
- Escalation: a `HasConfidence`-marked output that omits the required numeric `confidence` field now fails closed (halts, or escalates under a policy, with reason `confidence_marker_without_value`) instead of letting an `AttributeError` propagate out of `Pipeline.run()` — preserving the halt-and-escalate invariant.
- LLM gateway hardening: a model returning a 200 with an unexpected response shape (missing `choices`/`message`/`content`) now fails closed as a legible `LLMError` instead of escaping as a raw `KeyError`/`IndexError` — consistent with the non-JSON contract-violation path and the legibility principle. The non-object-JSON fail-closed branch is now test-covered.
- LLM halt-reason legibility: a model-backed agent whose fallback chain is exhausted now halts/escalates with the legible reason `model_unavailable` (and a model returning unusable content with `model_error`) instead of the generic `agent_error` — so an escalation reader can tell the LLM provider chain failed, not the agent's own code. `LLMError`/`ModelUnavailableError` self-declare a `halt_reason` class attribute, which `halt_reason_for` honours after the curated `_HALT_REASONS` registry — keeping `drawbore.errors` free of any `drawbore.llm` import (no cycle).
- Live MCP transport resource safety: `LiveMCPClient.connect` now closes the transport if `ClientSession` setup/`initialize` fails after the connection opened, `LiveMCPClient.close` tears the transport down even if the session exit raises (try/finally), and `register_mcp_server` closes the session if a declared tool is not advertised — so a failed connect/registration on a real server cannot leak a socket. (No effect on `FakeMCPClient`; the live path needs the `drawbore[mcp]` extra.)

### Documentation

- Fixed the remittance example snippet (missing required bindings) and the
  agentic-tool-loop snippet (undefined variable, superseded constructor form);
  added the risk-branch example page; normalized imports to the top-level surface.
- Agent guidance no longer references a partial-results option that does not exist.
- README rewritten to describe the implemented SDK (was a stale scaffolding notice).
