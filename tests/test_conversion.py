from typing import Any

import pytest

from asuka.contracts import Sample
from asuka.conversion import flatten_sample_groups, validate_sample_groups


def make_sample(
    response: str,
    *,
    rollout_id: int | None = None,
    group_id: int | None = None,
    sample_id: int | None = None,
) -> Sample:
    return Sample(
        prompt="solve 2 + 3",
        response=response,
        tokens=[1, 2, 3],
        response_length=1,
        rollout_id=rollout_id,
        group_id=group_id,
        sample_id=sample_id,
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
