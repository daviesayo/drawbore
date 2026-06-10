# Number chain

A three-agent live OpenAI Drawbore pipeline where each model-backed step receives
the previous step's typed output:

1. `draw_number` asks the model to create a random five-digit number.
2. `inspect_number` asks the model to check whether that number is prime and
   divisible by 5.
3. `explain_result` asks the model to turn those booleans into a useful
   explanation.

## Run

```bash
PYTHONPATH=src OPENAI_API_KEY=sk-... python3.11 examples/number_chain/pipeline.py
```

The script makes three real OpenAI calls, then prints each step's output and
`result.audit_trace.legible()`.

## Test

```bash
python3.11 -m pytest examples/number_chain/test_number_chain.py
```

The test checks the live-provider wiring without calling OpenAI.
