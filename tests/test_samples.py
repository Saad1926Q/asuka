import pytest

from asuka.data.contracts import SampleStatus
from asuka.rollout.samples import generation_to_sample
from asuka.rollout.sglang import GenerationResult


def make_result() -> GenerationResult:
    return GenerationResult(
        text=" answer",
        token_ids=[42, 7],
        log_probs=[-0.2, -0.4],
        meta_info={"finish_reason": "stop"},
    )


def test_generation_to_sample_builds_training_sample() -> None:
    sample = generation_to_sample(
        "question",
        [1, 2, 3],
        make_result(),
        group_id=10,
        rollout_id=20,
        sample_id=30,
        policy_version=4,
        metadata={"source": "test"},
    )

    assert sample.prompt == "question"
    assert sample.response == " answer"
    assert sample.tokens == [1, 2, 3, 42, 7]
    assert sample.response_length == 2
    assert sample.loss_mask == [1, 1]
    assert sample.rollout_log_probs == [-0.2, -0.4]
    assert sample.group_id == 10
    assert sample.rollout_id == 20
    assert sample.sample_id == 30
    assert sample.policy_version == 4
    assert sample.status is SampleStatus.COMPLETED
    assert sample.metadata == {"source": "test", "finish_reason": "stop"}


def test_generation_to_sample_copies_metadata() -> None:
    metadata = {"source": "test"}
    sample = generation_to_sample(
        "question",
        [1],
        make_result(),
        group_id=1,
        rollout_id=2,
        sample_id=3,
        metadata=metadata,
    )

    sample.metadata["new"] = True

    assert metadata == {"source": "test"}


def test_generation_to_sample_rejects_mismatched_generation_fields() -> None:
    result = GenerationResult(
        text="answer",
        token_ids=[42],
        log_probs=[],
        meta_info={},
    )

    with pytest.raises(ValueError, match="counts must match"):
        generation_to_sample(
            "question",
            [1],
            result,
            group_id=1,
            rollout_id=2,
            sample_id=3,
        )
