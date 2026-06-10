# Drawbore

> Composable agents that hold by construction.

Drawbore is a Python SDK for building constrained, composable agent pipelines
with safety guarantees baked in as architecture — for high-stakes, deterministic
workflows in fintech, healthcare, legal, compliance, and beyond.

The thesis: **capability is a liability in high-stakes workflows; constraint is
the feature.** You compose tightly scoped, typed, schema-enforced, audit-logged
agents; the framework enforces the contracts and you write the business logic.

## Install

```bash
pip install drawbore            # core
pip install "drawbore[mcp]"     # + MCP server tools
pip install "drawbore[otlp]"    # + OTLP span export
```

## Sixty seconds

```python
from pydantic import BaseModel
from drawbore import agent, Pipeline

class Payment(BaseModel):
    cents: int

class Normalized(BaseModel):
    dollars: float

@agent(name="normalize", input=Payment, output=Normalized)
async def normalize(p: Payment) -> Normalized:
    return Normalized(dollars=p.cents / 100)

pipeline = Pipeline(name="payments", version="1.0.0")
pipeline.add(normalize)

async def main():
    async with pipeline.test_mode() as tp:
        result = await tp.run(Payment(cents=2500))
    assert result.status == "completed"
    print(result.audit_trace.legible())   # a regulator-readable run record
```

Every step is schema-checked at both edges. Every tool call passes through a
proxy with a single-use permission token. Every run produces a legible audit
record. None of it is configurable off.

## What the framework guarantees

- **Typed contracts at every edge** — Pydantic-validated inputs and outputs;
  a wrong shape halts the run with a readable reason, it never propagates.
- **Least-privilege tools** — agents call only the tools they declared, through
  one audited chokepoint, with single-use scoped tokens and circuit breakers.
- **Halt-and-escalate by default** — failures stop the pipeline and produce an
  escalation package a human can read; there is no silent degradation.
- **Audit as a side effect** — `result.audit_trace.legible()` renders what ran,
  what was called, and why it stopped, with no tracing setup.
- **Test mode with the real safety layer** — `pipeline.test_mode(...)` mocks the
  outside world (models, tools, MCP) while every control runs for real.

## Learn more

- [Quickstart](docs/quickstart.mdx) — install to first run.
- [Guides](docs/guide/) — pipelines, tools, model-backed agents, escalation,
  testing, observability.
- [Examples](examples/) — runnable, tested pipelines from tiny to high-stakes.

## License

MIT
