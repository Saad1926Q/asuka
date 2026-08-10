"""Conversion helpers for rollout sample groups."""

from __future__ import annotations

from collections.abc import Sequence

from asuka.contracts import Sample


def validate_sample_groups(groups: Sequence[Sequence[Sample]]) -> None:
    """Validate grouped rollout output before flattening."""

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
    """Flatten grouped rollout samples while preserving order."""

    validate_sample_groups(groups)
    return [sample for group in groups for sample in group]
