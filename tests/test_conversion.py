from typing import Any

import pytest

from asuka.data.contracts import Sample
from asuka.data.conversion import (
    convert_samples_to_train_data,
    flatten_sample_groups,
    validate_sample_groups,
)


def make_sample(
    response: str,
    *,
    rollout_id: int | None = None,
    group_id: int | None = None,
    sample_id: int | None = None,
    reward: float | None = None,
    raw_reward: float | None = None,
    loss_mask: list[int] | None = None,
    rollout_log_probs: list[float] | None = None,
    policy_version: int = 0,
    metadata: dict[str, Any] | None = None,
    rollout_routed_experts: list[list[int]] | None = None,
) -> Sample:
    return Sample(
        prompt="solve 2 + 3",
        response=response,
        tokens=[1, 2, 3],
        response_length=1,
        rollout_id=rollout_id,
        group_id=group_id,
        sample_id=sample_id,
        reward=reward,
        raw_reward=raw_reward,
        loss_mask=loss_mask,
        rollout_log_probs=rollout_log_probs,
        policy_version=policy_version,
        metadata={} if metadata is None else metadata,
        rollout_routed_experts=rollout_routed_experts,
    )


def test_flatten_sample_groups_preserves_group_order() -> None:
    sample_a = make_sample("5", group_id=0, sample_id=0)
    sample_b = make_sample("6", group_id=0, sample_id=1)
    sample_c = make_sample("7", group_id=1, sample_id=2)

    flattened = flatten_sample_groups([[sample_a, sample_b], [sample_c]])

    assert flattened == [sample_a, sample_b, sample_c]


def test_validate_sample_groups_rejects_empty_outer_list() -> None:
    with pytest.raises(ValueError, match="at least one group"):
        validate_sample_groups([])


def test_validate_sample_groups_rejects_empty_inner_group() -> None:
    with pytest.raises(ValueError, match="group 0 is empty"):
        validate_sample_groups([[]])


def test_validate_sample_groups_rejects_non_sample_items() -> None:
    bad_groups: Any = [[object()]]

    with pytest.raises(TypeError, match="must be Sample"):
        validate_sample_groups(bad_groups)


def test_validate_sample_groups_accepts_multi_step_chunks_with_same_rollout_id() -> None:
    validate_sample_groups(
        [
            [
                make_sample("step 1", rollout_id=10, sample_id=0),
                make_sample("step 2", rollout_id=10, sample_id=1),
            ]
        ]
    )


def test_validate_sample_groups_rejects_missing_rollout_id_inside_multi_step_group() -> None:
    with pytest.raises(ValueError, match="mixes set and missing rollout_id"):
        validate_sample_groups(
            [
                [
                    make_sample("step 1", rollout_id=10, sample_id=0),
                    make_sample("step 2", rollout_id=None, sample_id=1),
                ]
            ]
        )


def test_validate_sample_groups_rejects_mismatched_rollout_ids_inside_group() -> None:
    with pytest.raises(ValueError, match="multiple rollout_id values"):
        validate_sample_groups(
            [
                [
                    make_sample("step 1", rollout_id=10, sample_id=0),
                    make_sample("step 2", rollout_id=11, sample_id=1),
                ]
            ]
        )


def test_convert_samples_to_train_data_collects_aligned_columns() -> None:
    sample_a = make_sample(
        "5",
        rollout_id=10,
        group_id=2,
        sample_id=0,
        reward=0.5,
        raw_reward=1.0,
        loss_mask=[1],
        rollout_log_probs=[-0.2],
        policy_version=3,
        metadata={"task": "math"},
        rollout_routed_experts=[[1, 4]],
    )
    sample_b = make_sample(
        "6",
        rollout_id=11,
        group_id=2,
        sample_id=1,
        reward=-0.5,
        raw_reward=0.0,
        loss_mask=[1],
        rollout_log_probs=[-0.4],
        policy_version=3,
        metadata={"task": "math"},
        rollout_routed_experts=[[2, 7]],
    )

    train_data = convert_samples_to_train_data([sample_a, sample_b])

    assert train_data.batch_size == 2
    assert train_data.tokens == [[1, 2, 3], [1, 2, 3]]
    assert train_data.response_lengths == [1, 1]
    assert train_data.rewards == [0.5, -0.5]
    assert train_data.raw_rewards == [1.0, 0.0]
    assert train_data.loss_masks == [[1], [1]]
    assert train_data.rollout_ids == [10, 11]
    assert train_data.group_ids == [2, 2]
    assert train_data.sample_ids == [0, 1]
    assert train_data.policy_versions == [3, 3]
    assert train_data.rollout_log_probs == [[-0.2], [-0.4]]
    assert train_data.rollout_routed_experts == [[[1, 4]], [[2, 7]]]
    assert train_data.metadata == [{"task": "math"}, {"task": "math"}]


def test_convert_samples_to_train_data_rejects_empty_sample_list() -> None:
    with pytest.raises(ValueError, match="empty sample list"):
        convert_samples_to_train_data([])


def test_convert_samples_to_train_data_rejects_missing_reward() -> None:
    with pytest.raises(ValueError, match="missing reward"):
        convert_samples_to_train_data([make_sample("5")])


def test_convert_samples_to_train_data_fills_default_loss_mask_and_raw_reward() -> None:
    sample = make_sample("5", reward=1.0)

    train_data = convert_samples_to_train_data([sample])

    assert train_data.loss_masks == [[1]]
    assert train_data.raw_rewards == [1.0]


def test_convert_samples_to_train_data_rejects_bad_loss_mask_length() -> None:
    sample = make_sample("5", reward=1.0, loss_mask=[1, 1])

    with pytest.raises(ValueError, match="loss_mask length"):
        convert_samples_to_train_data([sample])


def test_convert_samples_to_train_data_rejects_bad_logprob_length() -> None:
    sample = make_sample("5", reward=1.0, rollout_log_probs=[-0.1, -0.2])

    with pytest.raises(ValueError, match="rollout_log_probs length"):
        convert_samples_to_train_data([sample])


def test_convert_samples_to_train_data_rejects_bad_routed_experts_length() -> None:
    sample = make_sample("5", reward=1.0, rollout_routed_experts=[[1], [2]])

    with pytest.raises(ValueError, match="rollout_routed_experts length"):
        convert_samples_to_train_data([sample])


def test_convert_samples_to_train_data_assigns_missing_ids_deterministically() -> None:
    sample_a = make_sample("5", reward=1.0)
    sample_b = make_sample("6", reward=0.0)

    train_data = convert_samples_to_train_data([sample_a, sample_b])

    assert train_data.rollout_ids == [0, 1]
    assert train_data.group_ids == [0, 1]
    assert train_data.sample_ids == [0, 1]


def test_convert_samples_to_train_data_can_normalize_rewards_by_group() -> None:
    samples = [
        make_sample("5", reward=1.0, raw_reward=1.0, group_id=0),
        make_sample("6", reward=0.0, raw_reward=0.0, group_id=0),
        make_sample("7", reward=10.0, raw_reward=10.0, group_id=1),
        make_sample("8", reward=6.0, raw_reward=6.0, group_id=1),
    ]

    train_data = convert_samples_to_train_data(samples, normalize_rewards=True)

    assert train_data.rewards == [0.5, -0.5, 2.0, -2.0]
    assert train_data.raw_rewards == [1.0, 0.0, 10.0, 6.0]


def test_convert_samples_to_train_data_default_reward_behavior_is_unchanged() -> None:
    samples = [
        make_sample("5", reward=1.0, group_id=0),
        make_sample("6", reward=0.0, group_id=0),
    ]

    train_data = convert_samples_to_train_data(samples)

    assert train_data.rewards == [1.0, 0.0]
    assert train_data.raw_rewards == [1.0, 0.0]


def test_convert_samples_to_train_data_rejects_missing_group_id_when_normalizing() -> None:
    sample = make_sample("5", reward=1.0, group_id=None)

    with pytest.raises(ValueError, match="missing group_id"):
        convert_samples_to_train_data([sample], normalize_rewards=True)


def test_convert_samples_to_train_data_can_normalize_rewards_by_group_std() -> None:
    samples = [
        make_sample("5", reward=1.0, group_id=0),
        make_sample("6", reward=0.0, group_id=0),
    ]

    train_data = convert_samples_to_train_data(
        samples,
        normalize_rewards=True,
        normalize_rewards_by_std=True,
    )

    assert train_data.rewards == pytest.approx([0.999998, -0.999998])
