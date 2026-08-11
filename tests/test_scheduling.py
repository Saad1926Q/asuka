import pytest

from asuka.data.scheduling import DPSchedule, validate_dp_schedule


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
