"""Run Asuka's rollout-to-schedule flow against LFM2.5-350M."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from asuka.data.contracts import Sample
from asuka.data.pipeline import prepare_train_batch_from_rollouts
from asuka.data.scheduling import DPScheduleConfig
from asuka.rollout.runner import generate_batch
from asuka.rollout.sglang import SGLangClient

MODEL_ID = "LiquidAI/LFM2.5-350M"
SGLANG_URL = "http://localhost:30000"
SAMPLES_PER_PROMPT = 2
DP_SIZE = 2
MAX_NEW_TOKENS = 32

PROMPTS = {
    "What is 2 + 2? Reply with only the integer.": "4",
    "What is 3 + 5? Reply with only the integer.": "8",
    "What is 7 - 4? Reply with only the integer.": "3",
    "What is 6 + 1? Reply with only the integer.": "7",
}


def exact_match_reward(sample: Sample) -> float:
    """Returns one when the response matches its expected arithmetic answer."""

    expected = sample.metadata.get("answer")
    if not isinstance(expected, str):
        raise ValueError("sample metadata is missing string 'answer'")
    return float(sample.response.strip() == expected.strip())


def tokenize_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """Applies LFM2.5's chat template and returns prompt token IDs."""

    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    tokens = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(token) for token in tokens]


async def run() -> None:
    """Runs generation, rewards, TrainData conversion, and DP scheduling."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = list(PROMPTS)

    metadata = [{"answer": PROMPTS[prompt]} for prompt in prompts]

    async with SGLangClient(SGLANG_URL) as client:
        sample_groups = await generate_batch(
            client,
            prompts,
            tokenize=lambda prompt: tokenize_prompt(tokenizer, prompt),
            samples_per_prompt=SAMPLES_PER_PROMPT,
            reward_fn=exact_match_reward,
            generation_kwargs={
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.1,
                "top_p": 1.0,
                "top_k": 50,
            },
            prompt_metadata=metadata,
        )

    sample_count = len(prompts) * SAMPLES_PER_PROMPT

    config = DPScheduleConfig(
        dp_size=DP_SIZE,
        global_batch_size=sample_count,
        micro_batch_size=1,
    )
    prepared = prepare_train_batch_from_rollouts(sample_groups, config)

    report = {
        "model": MODEL_ID,
        "num_prompts": len(prompts),
        "samples_per_prompt": SAMPLES_PER_PROMPT,
        "num_samples": prepared.train_data.batch_size,
        "raw_rewards": prepared.train_data.raw_rewards,
        "rewards": prepared.train_data.rewards,
        "responses": [sample.response for sample in prepared.samples],
        "partitions": prepared.schedule.partitions,
        "micro_batch_indices": prepared.schedule.micro_batch_indices,
        "num_microbatches": prepared.schedule.num_microbatches,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
