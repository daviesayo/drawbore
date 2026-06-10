"""Minimal live OpenAI smoke example for Drawbore.

Run from the repository root:

    PYTHONPATH=src OPENAI_API_KEY=sk-... python3.11 examples/live_openai_smoke/pipeline.py

This intentionally calls a real provider. Tests for this example only verify the
runtime wiring; they do not call OpenAI.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Literal

from pydantic import BaseModel

from drawbore import Pipeline, agent
from drawbore.llm import LLMRuntime, LLMRuntimeConfig, ModelProfile, ModelTarget, ProviderConfig
from drawbore.orchestration import ADKEngine


COPY_PASTE_COMMAND = (
    "PYTHONPATH=src OPENAI_API_KEY=sk-... "
    "python3.11 examples/live_openai_smoke/pipeline.py"
)


class JudgmentRequest(BaseModel):
    claim: str


class Judgment(BaseModel):
    verdict: Literal["supported", "unsupported"]
    confidence: float
    rationale: str


@agent(
    name="judge_claim",
    input=JudgmentRequest,
    output=Judgment,
    model="profile:judgment",
    instructions=(
        "Assess whether the claim is supported by the supplied text. Keep the "
        "rationale short and return high confidence only for obvious claims."
    ),
)
async def judge_claim(req: JudgmentRequest) -> Judgment:
    ...  # model-backed one-shot agent; Drawbore does not run this body


def build_llm_config() -> LLMRuntimeConfig:
    """One-provider runtime policy: profile -> OpenAI target, key from env."""
    return LLMRuntimeConfig(
        profiles={
            "judgment": ModelProfile(
                targets=(ModelTarget(provider="openai", model="gpt-4o-mini"),),
            ),
        },
        providers={
            "openai": ProviderConfig(credential_env="OPENAI_API_KEY"),
        },
    )


def build_pipeline() -> Pipeline:
    pipeline = Pipeline(name="live_openai_smoke", version="1.0.0")
    pipeline.add(judge_claim)
    return pipeline


async def run_smoke() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running this live-provider smoke example.")

    pipeline = build_pipeline()
    runtime = LLMRuntime(config=build_llm_config())
    result = await pipeline.run(
        JudgmentRequest(claim="The sentence 'Drawbore uses typed outputs' is a claim."),
        engine=ADKEngine(llm_runtime=runtime),
    )

    print(json.dumps({
        "status": result.status,
        "output": result.outputs.get("judge_claim").model_dump() if result.outputs else None,
        "reason": result.reason,
        "audit": result.audit_trace.legible() if result.audit_trace else None,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(run_smoke())
