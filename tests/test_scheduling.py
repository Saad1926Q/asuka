import pytest

from asuka.data.contracts import TrainData
from asuka.data.scheduling import (
    DPSchedule,
    DPScheduleConfig,
    build_dp_schedule,
    validate_dp_schedule,
)


def test_validate_dp_schedule_accepts_valid_schedule() -> None:
    schedule = DPSchedule(
        partitions=[[0, 2, 4, 6], [1, 3, 5, 7]],
        micro_batch_indices=[[[0, 1], [2, 3]], [[0, 1], [2, 3]]],
        num_microbatches=[2],
        global_batch_sizes=[8],
    )

    validate_dp_schedule(schedule, batch_size=8, dp_size=2)


def test_validate_dp_schedule_rejects_non_positive_dp_size() -> None:
    schedule = DPSchedule(
        partitions=[],
        micro_batch_indices=[],
        num_microbatches=[],
        global_batch_sizes=[],
    )

    with pytest.raises(ValueError, match="dp_size must be positive"):
        validate_dp_schedule(schedule, batch_size=0, dp_size=0)


def test_validate_dp_schedule_rejects_wrong_partition_count() -> None:
    schedule = DPSchedule(
        partitions=[[0]],
        micro_batch_indices=[[[0]]],
        num_microbatches=[1],
        global_batch_sizes=[1],
    )

    with pytest.raises(ValueError, match="expected 2 partitions"):
        validate_dp_schedule(schedule, batch_size=1, dp_size=2)


def test_validate_dp_schedule_rejects_wrong_microbatch_rank_count() -> None:
    schedule = DPSchedule(
        partitions=[[0], [1]],
        micro_batch_indices=[[[0]]],
        num_microbatches=[1],
        global_batch_sizes=[2],
    )

    with pytest.raises(ValueError, match="expected 2 micro_batch_indices"):
        validate_dp_schedule(schedule, batch_size=2, dp_size=2)


def test_validate_dp_schedule_rejects_duplicate_global_sample_indices() -> None:
    schedule = DPSchedule(
        partitions=[[0], [0]],
        micro_batch_indices=[[[0]], [[0]]],
        num_microbatches=[1],
        global_batch_sizes=[2],
    )

    with pytest.raises(ValueError, match="assigned more than once"):
        validate_dp_schedule(schedule, batch_size=1, dp_size=2)


def test_validate_dp_schedule_rejects_out_of_range_global_sample_indices() -> None:
    schedule = DPSchedule(
        partitions=[[0], [2]],
        micro_batch_indices=[[[0]], [[0]]],
        num_microbatches=[1],
        global_batch_sizes=[2],
    )

    with pytest.raises(ValueError, match="out-of-range"):
        validate_dp_schedule(schedule, batch_size=2, dp_size=2)


def test_validate_dp_schedule_rejects_wrong_microbatch_count_for_rank() -> None:
    schedule = DPSchedule(
        partitions=[[0, 2], [1, 3]],
        micro_batch_indices=[[[0], [1]], [[0]]],
        num_microbatches=[2],
        global_batch_sizes=[4],
    )

    with pytest.raises(ValueError, match="rank 1 has 1 microbatches"):
        validate_dp_schedule(schedule, batch_size=4, dp_size=2)


def test_validate_dp_schedule_rejects_missing_local_microbatch_indices() -> None:
    schedule = DPSchedule(
        partitions=[[0, 2], [1, 3]],
        micro_batch_indices=[[[0], []], [[0], [1]]],
        num_microbatches=[2],
        global_batch_sizes=[4],
    )

    with pytest.raises(ValueError, match="must tile local partition"):
        validate_dp_schedule(schedule, batch_size=4, dp_size=2)


def test_validate_dp_schedule_rejects_reordered_local_microbatch_indices() -> None:
    schedule = DPSchedule(
        partitions=[[0, 2], [1, 3]],
        micro_batch_indices=[[[1], [0]], [[0], [1]]],
        num_microbatches=[2],
        global_batch_sizes=[4],
    )

    with pytest.raises(ValueError, match="must tile local partition"):
        validate_dp_schedule(schedule, batch_size=4, dp_size=2)


def make_train_data(rollout_ids: list[int]) -> TrainData:
    batch_size = len(rollout_ids)
    return TrainData(
        tokens=[[index] for index in range(batch_size)],
        response_lengths=[1] * batch_size,
        rewards=[0.0] * batch_size,
        raw_rewards=[0.0] * batch_size,
        loss_masks=[[1] for _ in range(batch_size)],
        rollout_ids=rollout_ids,
        group_ids=list(range(batch_size)),
        sample_ids=list(range(batch_size)),
        policy_versions=[0] * batch_size,
        rollout_log_probs=[None] * batch_size,
        rollout_routed_experts=[None] * batch_size,
        metadata=[{} for _ in range(batch_size)],
    )


def test_build_dp_schedule_static_single_step() -> None:
    train_data = make_train_data(list(range(8)))

    schedule = build_dp_schedule(
        train_data,
        DPScheduleConfig(dp_size=2, global_batch_size=8, micro_batch_size=2),
    )

    assert schedule.partitions == [[0, 1, 4, 5], [2, 3, 6, 7]]
    assert schedule.micro_batch_indices == [[[0, 1], [2, 3]], [[0, 1], [2, 3]]]
    assert schedule.num_microbatches == [2]
    assert schedule.global_batch_sizes == [8]


def test_build_dp_schedule_static_two_steps() -> None:
    train_data = make_train_data(list(range(8)))

    schedule = build_dp_schedule(
        train_data,
        DPScheduleConfig(dp_size=2, global_batch_size=4, micro_batch_size=2),
    )

    assert schedule.partitions == [[0, 1, 4, 5], [2, 3, 6, 7]]
    assert schedule.micro_batch_indices == [[[0, 1], [2, 3]], [[0, 1], [2, 3]]]
    assert schedule.num_microbatches == [1, 1]
    assert schedule.global_batch_sizes == [4, 4]


def test_build_dp_schedule_keeps_multi_sample_rollouts_in_same_step() -> None:
    train_data = make_train_data([10, 10, 11, 11, 12, 12, 13, 13])

    schedule = build_dp_schedule(
        train_data,
        DPScheduleConfig(dp_size=2, global_batch_size=2, micro_batch_size=2),
    )

    assert schedule.partitions == [[0, 1, 4, 5], [2, 3, 6, 7]]
    assert schedule.num_microbatches == [1, 1]
    assert schedule.global_batch_sizes == [2, 2]


def test_build_dp_schedule_rejects_invalid_config_values() -> None:
    train_data = make_train_data([0, 1])

    with pytest.raises(ValueError, match="dp_size must be positive"):
        build_dp_schedule(
            train_data, DPScheduleConfig(dp_size=0, global_batch_size=1, micro_batch_size=1)
        )
    with pytest.raises(ValueError, match="global_batch_size must be positive"):
        build_dp_schedule(
            train_data, DPScheduleConfig(dp_size=1, global_batch_size=0, micro_batch_size=1)
        )
    with pytest.raises(ValueError, match="micro_batch_size must be positive"):
        build_dp_schedule(
            train_data, DPScheduleConfig(dp_size=1, global_batch_size=1, micro_batch_size=0)
        )


def test_build_dp_schedule_rejects_unsupported_knobs_for_now() -> None:
    train_data = make_train_data([0, 1])

    with pytest.raises(NotImplementedError, match="dynamic token batching"):
        build_dp_schedule(
            train_data,
            DPScheduleConfig(
                dp_size=1,
                global_batch_size=1,
                micro_batch_size=1,
                use_dynamic_batch_size=True,
            ),
        )
    with pytest.raises(NotImplementedError, match="rank balancing"):
        build_dp_schedule(
            train_data,
            DPScheduleConfig(
                dp_size=1,
                global_batch_size=1,
                micro_batch_size=1,
                balance_data=True,
            ),
        )
    with pytest.raises(NotImplementedError, match="FLOPs balancing"):
        build_dp_schedule(
            train_data,
            DPScheduleConfig(
                dp_size=1,
                global_batch_size=1,
                micro_batch_size=1,
                balance_by_flops=True,
            ),
        )


def test_build_dp_schedule_rejects_incomplete_global_batch() -> None:
    train_data = make_train_data([0, 1, 2])

    with pytest.raises(ValueError, match="must be divisible by global_batch_size"):
        build_dp_schedule(
            train_data,
            DPScheduleConfig(dp_size=1, global_batch_size=2, micro_batch_size=1),
        )


def test_build_dp_schedule_rejects_incomplete_static_microbatch() -> None:
    train_data = make_train_data([0, 1, 2])

    with pytest.raises(ValueError, match="must be divisible by micro_batch_size"):
        build_dp_schedule(
            train_data,
            DPScheduleConfig(dp_size=1, global_batch_size=3, micro_batch_size=2),
        )


def test_build_dp_schedule_rejects_microbatch_count_not_divisible_by_dp_size() -> None:
    train_data = make_train_data([0, 1, 2])

    with pytest.raises(ValueError, match="not divisible by dp_size"):
        build_dp_schedule(
            train_data,
            DPScheduleConfig(dp_size=2, global_batch_size=3, micro_batch_size=1),
        )
