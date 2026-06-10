# Remittance validation

A high-stakes Drawbore pipeline: confirm a cross-border remittance. It fetches a
transaction through a narrow tool, scores its risk with a confidence-gated model
agent, and writes a typed confirmation — halting to a human when the model's
confidence is low.

This is a simplified, illustrative example. The full canonical scenario (identity
checks, an evidence-retrieval loop, and JSON-config round-trip) is exercised in the
project's acceptance suite.

## What it demonstrates

- A tool-backed step that reaches a database only through a declared tool.
- A confidence-gated model step (a `HasConfidence` output) that escalates instead
  of guessing when confidence is low.
- `From(...)` fan-in into the final confirmation step.
- Running it all in `pipeline.test_mode(...)` with mocked tool and model responses.

## Run

```bash
pip install drawbore
pytest examples/remittance_validation/test_remittance_validation.py
```

The source is `pipeline.py`. For the concepts, see the model-backed-agents, tools,
and escalation guides under `docs/guide/`.
