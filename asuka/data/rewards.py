"""Reward transforms used before Samples become TrainData."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from asuka.data.contracts import Sample


def normalize_group_rewards(
    samples: Sequence[Sample],
    *,
    normalize_by_std: bool = False,
    eps: float = 1e-6,
) -> list[float]:
    """Subtracts each group mean, optionally divides by group std, preserving order."""

    if not samples:
        raise ValueError("cannot normalize rewards for an empty sample list")

    grouped_rewards: dict[int, list[float]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if sample.reward is None:
            raise ValueError(f"sample {index} is missing reward")
        if sample.group_id is None:
            raise ValueError(f"sample {index} is missing group_id")
        grouped_rewards[sample.group_id].append(sample.reward)

    group_stats: dict[int, tuple[float, float]] = {}
    for group_id, rewards in grouped_rewards.items():
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        std = math.sqrt(variance)
        group_stats[group_id] = (mean, std)

    normalized: list[float] = []
    for sample in samples:
        assert sample.reward is not None
        assert sample.group_id is not None
        mean, std = group_stats[sample.group_id]
        reward = sample.reward - mean
        if normalize_by_std:
            reward = 0.0 if std < eps else reward / (std + eps)
        normalized.append(reward)

    return normalized
