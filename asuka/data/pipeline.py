"""Prepare rewarded rollout samples for distributed training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from asuka.data.contracts import Sample, TrainData
from asuka.data.conversion import convert_samples_to_train_data, flatten_sample_groups
from asuka.data.scheduling import DPSchedule, DPScheduleConfig, build_dp_schedule


@dataclass(slots=True)
class PreparedTrainBatch:
    """Flattened samples, trainer data, and its DP execution schedule."""

    samples: list[Sample]
    train_data: TrainData
    schedule: DPSchedule


def prepare_train_batch_from_rollouts(
    sample_groups: Sequence[Sequence[Sample]],
    config: DPScheduleConfig,
    *,
    normalize_rewards: bool = True,
    normalize_rewards_by_std: bool = False,
) -> PreparedTrainBatch:
    """Converts rewarded rollout groups into scheduled training data.

    Args:
        sample_groups: Prompt groups whose samples have already been rewarded.
        config: Static/dynamic packing and DP scheduling configuration.
        normalize_rewards: Whether to center rewards within each prompt group.
        normalize_rewards_by_std: Whether to divide centered rewards by group std.

    Returns:
        Flattened samples, columnar TrainData, and the corresponding DP schedule.
    """

    samples = flatten_sample_groups(sample_groups)

    train_data = convert_samples_to_train_data(
        samples,
        normalize_rewards=normalize_rewards,
        normalize_rewards_by_std=normalize_rewards_by_std,
    )

    schedule = build_dp_schedule(train_data, config)

    return PreparedTrainBatch(samples=samples, train_data=train_data, schedule=schedule)
