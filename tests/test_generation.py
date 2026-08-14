import asyncio
from typing import Any

import pytest

from asuka.data.contracts import Sample
from asuka.data.rewards import normalize_group_rewards
from asuka.rollout.generation import generate_group
from asuka.rollout.sglang import GenerationResult


class FakeGenerationClient:
    def __init__(self) -> None:
        self.prompts: list[list[int]] = []

    async def generate(self, prompt_tokens: list[int], **_kwargs: Any) -> GenerationResult:
        self.prompts.append(prompt_tokens)
        token = 40 + len(self.prompts)
        return GenerationResult(
            text=str(token),
            token_ids=[token],
            log_probs=[-0.1 * token],
            meta_info={"finish_reason": "stop"},
        )


def test_generate_group_builds_and_scores_alternatives() -> None:
    client = FakeGenerationClient()

    async def run() -> list[Sample]:
        return await generate_group(
            client,
            "question",
            [1, 2],
            group_id=10,
            rollout_ids=[100, 101, 102],
            sample_ids=[200, 201, 202],
            reward_fn=lambda sample: 1.0 if sample.response != "42" else 0.0,
            generation_kwargs={"max_new_tokens": 4},
        )

    samples = asyncio.run(run())

    assert [sample.group_id for sample in samples] == [10, 10, 10]
    assert [sample.rollout_id for sample in samples] == [100, 101, 102]
    assert [sample.sample_id for sample in samples] == [200, 201, 202]
    assert [sample.reward for sample in samples] == [1.0, 0.0, 1.0]
    assert [sample.raw_reward for sample in samples] == [1.0, 0.0, 1.0]
    assert client.prompts == [[1, 2], [1, 2], [1, 2]]


def test_generate_group_supports_async_rewards() -> None:
    async def reward_fn(sample: Sample) -> float:
        await asyncio.sleep(0)
        return float(len(sample.response))

    async def run() -> list[Sample]:
        return await generate_group(
            FakeGenerationClient(),
            "question",
            [1],
            group_id=1,
            rollout_ids=[2],
            sample_ids=[3],
            reward_fn=reward_fn,
        )

    samples = asyncio.run(run())

    assert samples[0].reward == 2.0


def test_generate_group_preserves_alternative_order() -> None:
    async def run() -> list[Sample]:
        return await generate_group(
            FakeGenerationClient(),
            "question",
            [1],
            group_id=1,
            rollout_ids=[20, 10],
            sample_ids=[200, 100],
            reward_fn=lambda _sample: 0.0,
        )

    samples = asyncio.run(run())

    assert [sample.rollout_id for sample in samples] == [20, 10]
    assert [sample.sample_id for sample in samples] == [200, 100]


def test_generate_group_rejects_mismatched_ids() -> None:
    with pytest.raises(ValueError, match="same length"):
        asyncio.run(
            generate_group(
                FakeGenerationClient(),
                "question",
                [1],
                group_id=1,
                rollout_ids=[2, 3],
                sample_ids=[4],
                reward_fn=lambda _sample: 0.0,
            )
        )


def test_group_can_be_normalized_after_generation() -> None:
    async def run() -> list[Sample]:
        return await generate_group(
            FakeGenerationClient(),
            "question",
            [1],
            group_id=1,
            rollout_ids=[2, 3],
            sample_ids=[4, 5],
            reward_fn=lambda sample: 1.0 if sample.response == "41" else 0.0,
        )

    samples = asyncio.run(run())

    assert normalize_group_rewards(samples) == [0.5, -0.5]
