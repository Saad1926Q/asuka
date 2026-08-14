"""Generate and score one prompt group through a rollout backend."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol, cast

from asuka.data.contracts import Sample
from asuka.rollout.samples import generation_to_sample
from asuka.rollout.sglang import GenerationResult


class GenerationClient(Protocol):
    """Backend interface required by grouped generation."""

    async def generate(
        self,
        prompt_tokens: list[int],
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: list[str] | None = None,
        **extra_sampling_params: Any,
    ) -> GenerationResult:
        """Generates one response for the supplied prompt tokens."""

        ...


RewardFn = Callable[[Sample], float | Awaitable[float]]


async def generate_group(
    client: GenerationClient,
    prompt: str,
    prompt_tokens: list[int],
    *,
    group_id: int,
    rollout_ids: list[int],
    sample_ids: list[int],
    reward_fn: RewardFn,
    generation_kwargs: Mapping[str, Any] | None = None,
    policy_version: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> list[Sample]:
    """Generates and scores several alternatives for one prompt.

    Each alternative gets a distinct trajectory/sample ID, while all
    alternatives share ``group_id`` for group-level reward processing.

    Args:
        client: Async rollout backend such as :class:`SGLangClient`.
        prompt: Prompt text shared by every alternative in the group.
        prompt_tokens: Prompt token IDs shared by every generation request.
        group_id: ID shared by alternatives sampled from this prompt.
        rollout_ids: One trajectory ID for each requested alternative.
        sample_ids: One trainable sample ID for each requested alternative.
        reward_fn: Sync or async function that scores a completed Sample.
        generation_kwargs: Sampling options forwarded to the client.
        policy_version: Policy version used to generate the responses.
        metadata: Metadata copied onto every generated Sample.

    Returns:
        Completed, rewarded Samples in the same order as ``rollout_ids``.
    """

    if not rollout_ids:
        raise ValueError("at least one rollout ID is required")
    if len(rollout_ids) != len(sample_ids):
        raise ValueError("rollout_ids and sample_ids must have the same length")

    request_kwargs = dict(generation_kwargs or {})

    async def generate_one(rollout_id: int, sample_id: int) -> Sample:
        result = await client.generate(prompt_tokens, **request_kwargs)

        sample = generation_to_sample(
            prompt,
            prompt_tokens,
            result,
            group_id=group_id,
            rollout_id=rollout_id,
            sample_id=sample_id,
            policy_version=policy_version,
            metadata=metadata,
        )

        reward = reward_fn(sample)

        if inspect.isawaitable(reward):
            reward = await reward

        score = float(cast(float, reward))

        sample.reward = score

        sample.raw_reward = score

        return sample

    return list(
        await asyncio.gather(
            *(
                generate_one(rollout_id, sample_id)
                for rollout_id, sample_id in zip(rollout_ids, sample_ids, strict=True)
            )
        )
    )
