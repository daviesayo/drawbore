# Live OpenAI smoke

A minimal one-step Drawbore pipeline that calls OpenAI through the current
`LLMRuntime` path. It proves the production provider wiring end to end:

- the agent declares `model="profile:judgment"`;
- the runtime profile binds that capability to OpenAI;
- the OpenAI key is read from `OPENAI_API_KEY`;
- the pipeline validates the model output before reporting success.

## Run

```bash
PYTHONPATH=src OPENAI_API_KEY=sk-... python3.11 examples/live_openai_smoke/pipeline.py
```

This command makes a real provider call and may incur provider cost. The test for
this example checks the wiring only:

```bash
python3.11 -m pytest examples/live_openai_smoke/test_live_openai_smoke.py
```
