"""Validate grouped rollout output and assemble trainer-facing batches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from asuka.data.contracts import Sample, TrainData
from asuka.data.rewards import normalize_group_rewards


def validate_sample_groups(groups: Sequence[Sequence[Sample]]) -> None:
    """Checks rollout groups are non-empty Samples with consistent rollout ids."""

    if not groups:
        raise ValueError("rollout output must contain at least one group")

    for group_index, group in enumerate(groups):
        if not group:
            raise ValueError(f"rollout group {group_index} is empty")

        for sample_index, sample in enumerate(group):
            if not isinstance(sample, Sample):
                raise TypeError(
                    f"rollout group {group_index} item {sample_index} must be Sample, "
                    f"got {type(sample).__name__}"
                )

        rollout_ids = [sample.rollout_id for sample in group]
        has_any_rollout_id = any(rollout_id is not None for rollout_id in rollout_ids)
        if has_any_rollout_id and any(rollout_id is None for rollout_id in rollout_ids):
            raise ValueError(
                f"rollout group {group_index} mixes set and missing rollout_id values: "
                f"{rollout_ids}"
            )

        set_rollout_ids = {rollout_id for rollout_id in rollout_ids if rollout_id is not None}
        if len(set_rollout_ids) > 1:
            raise ValueError(
                f"rollout group {group_index} has multiple rollout_id values: {rollout_ids}"
            )


def flatten_sample_groups(groups: Sequence[Sequence[Sample]]) -> list[Sample]:
    """Converts list[list[Sample]] into list[Sample] after validation."""

    validate_sample_groups(groups)
    return [sample for group in groups for sample in group]


def convert_samples_to_train_data(
    samples: Sequence[Sample],
    *,
    normalize_rewards: bool = False,
    normalize_rewards_by_std: bool = False,
) -> TrainData:
    """Builds TrainData columns, filling masks/ids and optionally normalizing rewards."""

    if not samples:
        raise ValueError("cannot convert an empty sample list to TrainData")

    processed_rewards = (
        normalize_group_rewards(samples, normalize_by_std=normalize_rewards_by_std)
        if normalize_rewards
        else None
    )

    tokens: list[list[int]] = []
    response_lengths: list[int] = []
    rewards: list[float] = []
    raw_rewards: list[float] = []
    loss_masks: list[list[int]] = []
    rollout_ids: list[int] = []
    group_ids: list[int] = []
    sample_ids: list[int] = []
    policy_versions: list[int] = []
    rollout_log_probs: list[list[float] | None] = []
    rollout_routed_experts: list[list[list[int]] | None] = []
    metadata: list[dict[str, Any]] = []

    for index, sample in enumerate(samples):
        if sample.reward is None:
            raise ValueError(f"sample {index} is missing reward")

        reward = processed_rewards[index] if processed_rewards is not None else sample.reward

        mask = sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length
        if len(mask) != sample.response_length:
            raise ValueError(
                f"sample {index} loss_mask length {len(mask)} does not match "
                f"response_length {sample.response_length}"
            )

        if (
            sample.rollout_log_probs is not None
            and len(sample.rollout_log_probs) != sample.response_length
        ):
            raise ValueError(
                f"sample {index} rollout_log_probs length {len(sample.rollout_log_probs)} "
                f"does not match response_length {sample.response_length}"
            )

        if (
            sample.rollout_routed_experts is not None
            and len(sample.rollout_routed_experts) != sample.response_length
        ):
            raise ValueError(
                f"sample {index} rollout_routed_experts length "
                f"{len(sample.rollout_routed_experts)} does not match response_length "
                f"{sample.response_length}"
            )

        tokens.append(sample.tokens)
        response_lengths.append(sample.response_length)
        rewards.append(reward)
        raw_rewards.append(sample.raw_reward if sample.raw_reward is not None else sample.reward)
        loss_masks.append(mask)
        rollout_ids.append(sample.rollout_id if sample.rollout_id is not None else index)
        group_ids.append(sample.group_id if sample.group_id is not None else index)
        sample_ids.append(sample.sample_id if sample.sample_id is not None else index)
        policy_versions.append(sample.policy_version)
        rollout_log_probs.append(sample.rollout_log_probs)
        rollout_routed_experts.append(sample.rollout_routed_experts)
        metadata.append(dict(sample.metadata))

    return TrainData(
        tokens=tokens,
        response_lengths=response_lengths,
        rewards=rewards,
        raw_rewards=raw_rewards,
        loss_masks=loss_masks,
        rollout_ids=rollout_ids,
        group_ids=group_ids,
        sample_ids=sample_ids,
        policy_versions=policy_versions,
        rollout_log_probs=rollout_log_probs,
        rollout_routed_experts=rollout_routed_experts,
        metadata=metadata,
    )
