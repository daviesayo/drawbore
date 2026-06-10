# Risk branch

A deterministic branch + join pipeline that shows how to gate steps on an upstream
field with `When` and merge mutually exclusive branches with an explicit `Join`.

The pipeline has four agents:

1. `risk` — classifies the transaction amount as `"high"` or `"low"`.
2. `enhanced_dd` — runs only when `risk.level == "high"` (enhanced due diligence).
3. `standard_review` — runs only when `risk.level in ["low", "medium"]`.
4. `writer` — receives the result of whichever review branch ran.

An `exactly_one` `Join` named `"review"` sits between the two branches and the
writer. It enforces that exactly one branch ran, then forwards its output. Every
skipped branch is recorded in the audit trace with the condition that was not met.

## Run

```bash
pytest examples/risk_branch -v
```

No provider or API key is needed — all agents are deterministic.

## Learn more

See the [Pipelines guide](../../docs/guide/pipelines.mdx) for the full
`When` / `Join` reference.
