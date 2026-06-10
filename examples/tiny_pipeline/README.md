# Tiny pipeline

The smallest useful Drawbore pipeline: two deterministic, typed steps. It
normalizes a payment amount, then categorizes it — no model, no tools.

## Run

```bash
pip install drawbore
pytest examples/tiny_pipeline/test_tiny_pipeline.py
```

`pipeline.test_mode()` runs the real safety layer; this pipeline needs no mocks
because it is fully deterministic.

The source is `pipeline.py`. For the concepts, see the Pipelines guide
(`docs/guide/pipelines.mdx`).
