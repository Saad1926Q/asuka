import pytest

from asuka.contracts import Sample, SampleStatus, TrainData


def test_sample_defaults_are_slime_like() -> None:
    sample = Sample(
        prompt="2 + 2 = ?",
        response="4",
        tokens=[101, 102, 103],
        response_length=1,
    )

    assert sample.reward is None
    assert sample.raw_reward is None
    assert sample.loss_mask is None
    assert sample.rollout_log_probs is None
    assert sample.rollout_id is None
    assert sample.group_id is None
    assert sample.sample_id is None
    assert sample.policy_version == 0
    assert sample.status is SampleStatus.PENDING
    assert sample.metadata == {}
    assert sample.rollout_routed_experts is None


def test_sample_can_store_rollout_training_and_moe_fields() -> None:
    sample = Sample(
        prompt="2 + 2 = ?",
        response="4",
        tokens=[10, 20, 30],
        response_length=1,
        reward=0.5,
        raw_reward=1.0,
        loss_mask=[1],
        rollout_log_probs=[-0.1],
        rollout_id=7,
        group_id=3,
        sample_id=11,
        policy_version=5,
        status=SampleStatus.COMPLETED,
        metadata={"source": "arith"},
        rollout_routed_experts=[[2, 5]],
    )

    assert sample.reward == 0.5
    assert sample.raw_reward == 1.0
    assert sample.loss_mask == [1]
    assert sample.rollout_log_probs == [-0.1]
    assert sample.rollout_id == 7
    assert sample.group_id == 3
    assert sample.sample_id == 11
    assert sample.policy_version == 5
    assert sample.status is SampleStatus.COMPLETED
    assert sample.metadata == {"source": "arith"}
    assert sample.rollout_routed_experts == [[2, 5]]


def test_sample_status_values_match_rollout_lifecycle_strings() -> None:
    assert [status.value for status in SampleStatus] == [
        "pending",
        "completed",
        "truncated",
        "aborted",
        "failed",
    ]


def test_train_data_batch_size_and_equal_length_fields() -> None:
    train_data = TrainData(
        tokens=[[1, 2, 3], [4, 5]],
        response_lengths=[1, 1],
        rewards=[1.0, 0.0],
        raw_rewards=[1.0, 0.0],
        loss_masks=[[1], [1]],
        rollout_ids=[0, 1],
        group_ids=[0, 0],
        sample_ids=[0, 1],
        policy_versions=[2, 2],
        rollout_log_probs=[[-0.2], None],
        rollout_routed_experts=[[[0, 1]], None],
        metadata=[{"source": "arith"}, {"source": "arith"}],
    )

    assert train_data.batch_size == 2


def test_train_data_rejects_mismatched_batch_lengths() -> None:
    with pytest.raises(ValueError, match="same batch length"):
        TrainData(
            tokens=[[1, 2, 3], [4, 5]],
            response_lengths=[1],
            rewards=[1.0, 0.0],
            raw_rewards=[1.0, 0.0],
            loss_masks=[[1], [1]],
            rollout_ids=[0, 1],
            group_ids=[0, 0],
            sample_ids=[0, 1],
            policy_versions=[2, 2],
            rollout_log_probs=[[-0.2], None],
            rollout_routed_experts=[[[0, 1]], None],
            metadata=[{}, {}],
        )
