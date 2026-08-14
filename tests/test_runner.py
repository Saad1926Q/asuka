import asyncio
from typing import Any

import pytest

from asuka.rollout.runner import generate_batch
from asuka.rollout.sglang import GenerationResult


class FakeBatchClient:
    async def generate(self, prompt_tokens: list[int], **_kwargs: Any) -> GenerationResult:
        token = sum(prompt_tokens)
        return GenerationResult(
            text=str(token),
            token_ids=[token],
            log_probs=[-0.1],
            meta_info={},
        )


def test_generate_batch_tokenizes_prompts_and_assigns_ids() -> None:
    async def run():
        return await generate_batch(
            FakeBatchClient(),
            ["one", "two"],
            tokenize=lambda prompt: [len(prompt)],
            samples_per_prompt=2,
            reward_fn=lambda sample: float(len(sample.response)),
            group_id_start=10,
            rollout_id_start=100,
            sample_id_start=200,
        )

    groups = asyncio.run(run())

    assert len(groups) == 2
    assert [[sample.group_id for sample in group] for group in groups] == [[10, 10], [11, 11]]
    assert [[sample.rollout_id for sample in group] for group in groups] == [
        [100, 101],
        [102, 103],
    ]
    assert [[sample.sample_id for sample in group] for group in groups] == [
        [200, 201],
        [202, 203],
    ]
    assert [[sample.tokens for sample in group] for group in groups] == [
        [[3, 3], [3, 3]],
        [[3, 3], [3, 3]],
    ]


def test_generate_batch_forwards_prompt_metadata() -> None:
    async def run():
        return await generate_batch(
            FakeBatchClient(),
            ["one", "two"],
            tokenize=lambda _prompt: [1],
            samples_per_prompt=1,
            reward_fn=lambda _sample: 1.0,
            prompt_metadata=[{"answer": "one"}, {"answer": "two"}],
        )

    groups = asyncio.run(run())

    assert groups[0][0].metadata["answer"] == "one"
    assert groups[1][0].metadata["answer"] == "two"


def test_generate_batch_rejects_mismatched_prompt_metadata() -> None:
    with pytest.raises(ValueError, match="prompt_metadata must match"):
        asyncio.run(
            generate_batch(
                FakeBatchClient(),
                ["one", "two"],
                tokenize=lambda _prompt: [1],
                samples_per_prompt=1,
                reward_fn=lambda _sample: 1.0,
                prompt_metadata=[{}],
            )
        )


def test_generate_batch_rejects_non_positive_sample_count() -> None:
    with pytest.raises(ValueError, match="samples_per_prompt must be positive"):
        asyncio.run(
            generate_batch(
                FakeBatchClient(),
                ["one"],
                tokenize=lambda _prompt: [1],
                samples_per_prompt=0,
                reward_fn=lambda _sample: 1.0,
            )
        )
