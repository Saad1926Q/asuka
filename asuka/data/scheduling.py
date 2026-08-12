"""Split trainer batches across data-parallel ranks and microbatches."""

from __future__ import annotations

from dataclasses import dataclass

from asuka.data.contracts import TrainData
from asuka.data.packing import expand_bins_by_splitting, first_fit_pack


@dataclass(slots=True)
class DPScheduleConfig:
    """Configuration that chooses batch size, rank count, and scheduling modes."""

    dp_size: int
    global_batch_size: int
    micro_batch_size: int
    use_dynamic_batch_size: bool = False
    max_tokens_per_rank: int | None = None
    balance_data: bool = False
    balance_by_flops: bool = False


@dataclass(slots=True)
class DPSchedule:
    """Rank partitions plus local microbatch order for one TrainData batch."""

    partitions: list[list[int]]
    micro_batch_indices: list[list[list[int]]]
    num_microbatches: list[int]
    global_batch_sizes: list[int]


def _validate_config(config: DPScheduleConfig) -> None:
    """Rejects impossible schedule sizes before building rank assignments."""

    if config.dp_size <= 0:
        raise ValueError("dp_size must be positive")
    if config.global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    if config.micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")


def _group_samples_by_rollout_id(rollout_ids: list[int]) -> dict[int, list[int]]:
    """Collects sample positions for each rollout id while preserving order."""

    grouped: dict[int, list[int]] = {}
    for sample_index, rollout_id in enumerate(rollout_ids):
        grouped.setdefault(rollout_id, []).append(sample_index)
    return grouped


def _make_static_microbatches(
    sample_indices: list[int],
    *,
    micro_batch_size: int,
) -> list[list[int]]:
    """Splits one training step into fixed-size sample-count microbatches."""

    if len(sample_indices) % micro_batch_size != 0:
        raise ValueError(
            f"step sample count ({len(sample_indices)}) must be divisible by "
            f"micro_batch_size ({micro_batch_size})"
        )
    return [
        sample_indices[start : start + micro_batch_size]
        for start in range(0, len(sample_indices), micro_batch_size)
    ]


def build_dp_schedule(train_data: TrainData, config: DPScheduleConfig) -> DPSchedule:
    """Builds the rank/microbatch plan for one TrainData batch.

    The schedule answers two questions: which samples go to each data-parallel
    rank, and how each rank splits its local samples into microbatches.

    Static mode groups a fixed number of samples using ``micro_batch_size``.
    Dynamic mode greedily packs samples by token length, placing each sample in
    the first existing microbatch where it fits under ``max_tokens_per_rank``;
    oversized samples are placed alone.

    Rank workload balancing and FLOPs-aware balancing are not implemented yet.
    """

    _validate_config(config)
    if config.balance_data:
        raise NotImplementedError("rank balancing is not implemented yet")
    if config.balance_by_flops:
        raise NotImplementedError("FLOPs balancing is not implemented yet")

    rollout_id_to_samples = _group_samples_by_rollout_id(train_data.rollout_ids)
    rollout_ids = list(rollout_id_to_samples)

    if len(rollout_ids) % config.global_batch_size != 0:
        raise ValueError(
            f"number of rollouts ({len(rollout_ids)}) must be divisible by "
            f"global_batch_size ({config.global_batch_size})"
        )

    partitions: list[list[int]] = [[] for _ in range(config.dp_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(config.dp_size)]
    num_microbatches: list[int] = []
    global_batch_sizes: list[int] = []

    for step_start in range(0, len(rollout_ids), config.global_batch_size):
        step_rollout_ids = rollout_ids[step_start : step_start + config.global_batch_size]

        step_sample_indices = [
            sample_index
            for rollout_id in step_rollout_ids
            for sample_index in rollout_id_to_samples[rollout_id]
        ]

        if config.use_dynamic_batch_size:
            if config.max_tokens_per_rank is None:
                raise ValueError("max_tokens_per_rank is required for dynamic token batching")

            lengths = [len(train_data.tokens[index]) for index in step_sample_indices]

            local_microbatches = first_fit_pack(
                lengths,
                config.max_tokens_per_rank,
            )

            target_count = (
                (len(local_microbatches) + config.dp_size - 1) // config.dp_size
            ) * config.dp_size
            if target_count > len(local_microbatches):
                expand_bins_by_splitting(
                    local_microbatches,
                    target_count,
                    lengths,
                )
            step_microbatches = [
                [step_sample_indices[index] for index in microbatch]
                for microbatch in local_microbatches
            ]
        else:
            step_microbatches = _make_static_microbatches(
                step_sample_indices,
                micro_batch_size=config.micro_batch_size,
            )

        if len(step_microbatches) % config.dp_size != 0:
            raise ValueError(
                f"step has {len(step_microbatches)} microbatches, which is not "
                f"divisible by dp_size ({config.dp_size})"
            )

        num_microbatches.append(len(step_microbatches) // config.dp_size)
        global_batch_sizes.append(config.global_batch_size)

        for rank in range(config.dp_size):
            for microbatch_index in range(rank, len(step_microbatches), config.dp_size):
                microbatch = step_microbatches[microbatch_index]
                local_start = len(partitions[rank])
                partitions[rank].extend(microbatch)
                micro_batch_indices[rank].append(
                    list(range(local_start, local_start + len(microbatch)))
                )

    schedule = DPSchedule(
        partitions=partitions,
        micro_batch_indices=micro_batch_indices,
        num_microbatches=num_microbatches,
        global_batch_sizes=global_batch_sizes,
    )
    validate_dp_schedule(schedule, batch_size=train_data.batch_size, dp_size=config.dp_size)
    return schedule


def validate_dp_schedule(schedule: DPSchedule, *, batch_size: int, dp_size: int) -> None:
    """Checks rank counts, sample ownership, and local microbatch coverage."""

    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if dp_size <= 0:
        raise ValueError("dp_size must be positive")
    if len(schedule.partitions) != dp_size:
        raise ValueError(f"expected {dp_size} partitions, got {len(schedule.partitions)}")
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
