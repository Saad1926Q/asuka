import pytest

from asuka.contracts import Sample
from asuka.rewards import normalize_group_rewards


def make_sample(reward: float | None, *, group_id: int | None) -> Sample:
    return Sample(
        prompt="solve 2 + 3",
        response="5",
        tokens=[1, 2, 3],
        response_length=1,
        reward=reward,
        group_id=group_id,
    )


def test_normalize_group_rewards_subtracts_group_mean() -> None:
    samples = [
        make_sample(1.0, group_id=0),
        make_sample(0.0, group_id=0),
        make_sample(1.0, group_id=0),
        make_sample(0.0, group_id=0),
    ]

    assert normalize_group_rewards(samples) == [0.5, -0.5, 0.5, -0.5]


def test_normalize_group_rewards_handles_groups_independently() -> None:
    samples = [
        make_sample(1.0, group_id=0),
        make_sample(0.0, group_id=0),
        make_sample(10.0, group_id=1),
        make_sample(6.0, group_id=1),
    ]

    assert normalize_group_rewards(samples) == [0.5, -0.5, 2.0, -2.0]


def test_normalize_group_rewards_can_divide_by_group_std() -> None:
    samples = [
        make_sample(1.0, group_id=0),
        make_sample(0.0, group_id=0),
    ]

    rewards = normalize_group_rewards(samples, normalize_by_std=True, eps=0.0)

    assert rewards == [1.0, -1.0]


def test_normalize_group_rewards_single_sample_group_becomes_zero() -> None:
    samples = [make_sample(1.0, group_id=0)]

    assert normalize_group_rewards(samples, normalize_by_std=True) == [0.0]


def test_normalize_group_rewards_rejects_empty_sample_list() -> None:
    with pytest.raises(ValueError, match="empty sample list"):
        normalize_group_rewards([])


def test_normalize_group_rewards_rejects_missing_reward() -> None:
    with pytest.raises(ValueError, match="missing reward"):
        normalize_group_rewards([make_sample(None, group_id=0)])


def test_normalize_group_rewards_rejects_missing_group_id() -> None:
    with pytest.raises(ValueError, match="missing group_id"):
        normalize_group_rewards([make_sample(1.0, group_id=None)])
