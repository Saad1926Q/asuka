from asuka.data.contracts import Sample, SampleStatus
from asuka.data.pipeline import prepare_train_batch_from_rollouts
from asuka.data.scheduling import DPScheduleConfig


def make_sample(group_id: int, sample_id: int, reward: float) -> Sample:
    return Sample(
        prompt=f"prompt {group_id}",
        response=str(reward),
        tokens=[1],
        response_length=1,
        reward=reward,
        raw_reward=reward,
        loss_mask=[1],
        rollout_id=sample_id,
        group_id=group_id,
        sample_id=sample_id,
        status=SampleStatus.COMPLETED,
    )


def test_prepare_train_batch_connects_conversion_and_schedule() -> None:
    prepared = prepare_train_batch_from_rollouts(
        [
            [make_sample(10, 100, 1.0), make_sample(10, 101, 0.0)],
            [make_sample(11, 102, 0.0), make_sample(11, 103, 1.0)],
        ],
        DPScheduleConfig(
            dp_size=2,
            global_batch_size=4,
            micro_batch_size=1,
        ),
    )

    assert [sample.sample_id for sample in prepared.samples] == [100, 101, 102, 103]
    assert prepared.train_data.rewards == [0.5, -0.5, -0.5, 0.5]
    assert prepared.schedule.partitions == [[0, 2], [1, 3]]
    assert prepared.schedule.micro_batch_indices == [
        [[0], [1]],
        [[0], [1]],
    ]


def test_prepare_train_batch_can_skip_reward_normalization() -> None:
    prepared = prepare_train_batch_from_rollouts(
        [[make_sample(10, 100, 1.0), make_sample(10, 101, 0.0)]],
        DPScheduleConfig(dp_size=1, global_batch_size=2, micro_batch_size=1),
        normalize_rewards=False,
    )

    assert prepared.train_data.rewards == [1.0, 0.0]
