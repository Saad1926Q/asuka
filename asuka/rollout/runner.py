"""Run grouped rollouts for a batch of prompts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from asuka.data.contracts import Sample
from asuka.rollout.generation import GenerationClient, RewardFn, generate_group


async def generate_batch(
    client: GenerationClient,
    prompts: Sequence[str],
    *,
    tokenize: Callable[[str], list[int]],
    samples_per_prompt: int,
    reward_fn: RewardFn,
    group_id_start: int = 0,
    rollout_id_start: int = 0,
    sample_id_start: int = 0,
    generation_kwargs: Mapping[str, Any] | None = None,
    prompt_metadata: Sequence[Mapping[str, Any] | None] | None = None,
) -> list[list[Sample]]:
    """Generates and scores a group of alternatives for every prompt.

    Args:
        client: Async rollout backend such as :class:`SGLangClient`.
        prompts: Prompt texts to generate responses for.
        tokenize: Function converting one prompt text into token IDs.
        samples_per_prompt: Number of alternatives to generate per prompt.
        reward_fn: Sync or async function that scores each completed Sample.
        group_id_start: First group ID assigned to the prompt batch.
        rollout_id_start: First trajectory ID assigned to the batch.
        sample_id_start: First trainable sample ID assigned to the batch.
        generation_kwargs: Sampling options forwarded to SGLang.
        prompt_metadata: Optional metadata aligned with ``prompts``.

    Returns:
        One list of rewarded Samples per prompt, preserving prompt and
        alternative order.
    """

    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive")

    if prompt_metadata is not None and len(prompt_metadata) != len(prompts):
        raise ValueError("prompt_metadata must match prompts in length")

    async def generate_prompt(prompt_index: int, prompt: str) -> list[Sample]:

        metadata = None if prompt_metadata is None else prompt_metadata[prompt_index]

        first_rollout_id = rollout_id_start + prompt_index * samples_per_prompt
        first_sample_id = sample_id_start + prompt_index * samples_per_prompt

        return await generate_group(
            client,
            prompt,
            tokenize(prompt),
            group_id=group_id_start + prompt_index,
            rollout_ids=list(range(first_rollout_id, first_rollout_id + samples_per_prompt)),
            sample_ids=list(range(first_sample_id, first_sample_id + samples_per_prompt)),
            reward_fn=reward_fn,
            generation_kwargs=generation_kwargs,
            metadata=metadata,
        )

    return list(
        await asyncio.gather(
            *(generate_prompt(index, prompt) for index, prompt in enumerate(prompts))
        )
    )
