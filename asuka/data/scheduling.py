"""Data-parallel schedule contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DPSchedule:
    """How a TrainData batch is assigned to DP ranks and microbatches."""

    partitions: list[list[int]]
    micro_batch_indices: list[list[list[int]]]
    num_microbatches: list[int]
    global_batch_sizes: list[int]


def validate_dp_schedule(schedule: DPSchedule, *, batch_size: int, dp_size: int) -> None:
    """Validate basic DP schedule invariants."""

    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if dp_size <= 0:
        raise ValueError("dp_size must be positive")
    if len(schedule.partitions) != dp_size:
        raise ValueError(
            f"expected {dp_size} partitions, got {len(schedule.partitions)}"
        )
    if len(schedule.micro_batch_indices) != dp_size:
        raise ValueError(
            f"expected {dp_size} micro_batch_indices entries, "
            f"got {len(schedule.micro_batch_indices)}"
        )

    expected_microbatch_count = sum(schedule.num_microbatches)
    seen_global_indices: set[int] = set()

    for rank, partition in enumerate(schedule.partitions):
        for sample_index in partition:
            if sample_index < 0 or sample_index >= batch_size:
                raise ValueError(
                    f"rank {rank} partition has out-of-range sample index {sample_index}"
                )
            if sample_index in seen_global_indices:
                raise ValueError(f"sample index {sample_index} is assigned more than once")
            seen_global_indices.add(sample_index)

        rank_microbatches = schedule.micro_batch_indices[rank]
        if len(rank_microbatches) != expected_microbatch_count:
            raise ValueError(
                f"rank {rank} has {len(rank_microbatches)} microbatches, "
                f"expected {expected_microbatch_count}"
            )

        local_indices = [index for microbatch in rank_microbatches for index in microbatch]
        expected_local_indices = list(range(len(partition)))
        if local_indices != expected_local_indices:
            raise ValueError(
                f"rank {rank} micro_batch_indices must tile local partition indices "
                f"{expected_local_indices}, got {local_indices}"
            )
