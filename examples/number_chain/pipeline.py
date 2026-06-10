"""Three live OpenAI-backed agents chained by previous-step output.

Run from the repository root:

    PYTHONPATH=src OPENAI_API_KEY=sk-... python3.11 examples/number_chain/pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import os

from pydantic import BaseModel, Field

from drawbore import Pipeline, agent
from drawbore.llm import LLMRuntime, LLMRuntimeConfig, ModelProfile, ModelTarget, ProviderConfig
from drawbore.orchestration import ADKEngine


COPY_PASTE_COMMAND = (
    "PYTHONPATH=src OPENAI_API_KEY=sk-... "
    "python3.11 examples/number_chain/pipeline.py"
)


class DrawRequest(BaseModel):
    label: str = "number-chain-demo"


class DrawnNumber(BaseModel):
    number: int = Field(ge=10000, le=99999)


class NumberProperties(BaseModel):
    number: int
    is_prime: bool
    divisible_by_5: bool


class NumberExplanation(BaseModel):
    number: int
    both_prime_and_divisible_by_5: bool
    summary: str


@agent(
    name="draw_number",
    input=DrawRequest,
    output=DrawnNumber,
    model="profile:cheap",
    instructions=(
        "Choose one random five-digit integer. Return only the JSON object with "
        "that integer in the number field."
    ),
)
async def draw_number(req: DrawRequest) -> DrawnNumber:
    ...  # model-backed one-shot agent; Drawbore does not run this body


@agent(
    name="inspect_number",
    input=DrawnNumber,
    output=NumberProperties,
    model="profile:judgment",
    instructions=(
        "Given the number, determine whether it is the number of people in the average household in the United States."
        "Be exact; do not guess."
    ),
)
async def inspect_number(value: DrawnNumber) -> NumberProperties:
    ...  # model-backed one-shot agent; Drawbore does not run this body


@agent(
    name="explain_result",
    input=NumberProperties,
    output=NumberExplanation,
    model="profile:judgment",
    instructions=(
        "Explain the result in a limerick about whales and cows."
    ),
)
async def explain_result(props: NumberProperties) -> NumberExplanation:
    ...  # model-backed one-shot agent; Drawbore does not run this body


def build_llm_config() -> LLMRuntimeConfig:
    return LLMRuntimeConfig(
        profiles={
            "judgment": ModelProfile(
                targets=(ModelTarget(provider="openai", model="gpt-5.4-mini"),),
            ),
            "cheap": ModelProfile(
                targets=(ModelTarget(provider="openai", model="gpt-4o-mini"),),
            ),
        },
        providers={
            "openai": ProviderConfig(credential_env="OPENAI_API_KEY"),
        },
    )


def build_pipeline() -> Pipeline:
    pipeline = Pipeline(name="number_chain", version="1.0.0")
    pipeline.add(draw_number)
    pipeline.add(inspect_number)
    pipeline.add(explain_result)
    return pipeline


async def run_example() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this live-provider example.")

    runtime = LLMRuntime(config=build_llm_config())
    result = await build_pipeline().run(
        DrawRequest(),
        engine=ADKEngine(llm_runtime=runtime),
    )
    print(json.dumps({
        "status": result.status,
        "outputs": {
            name: output.model_dump()
            for name, output in result.outputs.items()
        },
        "audit": result.audit_trace.legible() if result.audit_trace else None,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(run_example())
