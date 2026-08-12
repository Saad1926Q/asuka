import pytest

from asuka.data.packing import (
    expand_bins_by_splitting,
    first_fit_pack,
    split_bin_by_tokens,
)


def test_first_fit_pack_places_samples_in_first_bin_that_fits() -> None:
    assert first_fit_pack([1000, 1200, 3000, 900], 4096) == [[0, 1, 3], [2]]


def test_first_fit_pack_reuses_earlier_bin_before_opening_new_one() -> None:
    assert first_fit_pack([6, 4, 2, 5], 10) == [[0, 1], [2, 3]]


def test_first_fit_pack_places_oversized_sample_alone() -> None:
    assert first_fit_pack([5000, 1000, 1000], 4096) == [[0], [1, 2]]


def test_first_fit_pack_preserves_sample_order_within_bins() -> None:
    bins = first_fit_pack([3, 3, 3, 3], 6)

    assert bins == [[0, 1], [2, 3]]
    assert [index for bin_ in bins for index in bin_] == [0, 1, 2, 3]


def test_first_fit_pack_rejects_invalid_token_budget() -> None:
    with pytest.raises(ValueError, match="max_tokens_per_bin must be positive"):
        first_fit_pack([1], 0)


def test_first_fit_pack_rejects_negative_lengths() -> None:
    with pytest.raises(ValueError, match="lengths must be non-negative"):
        first_fit_pack([-1], 10)


def test_split_bin_by_tokens_balances_two_halves() -> None:
    left, right = split_bin_by_tokens([0, 1, 2], [100, 80, 20])

    assert left == [0]
    assert right == [1, 2]


def test_split_bin_by_tokens_rejects_singleton() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        split_bin_by_tokens([0], [10])


def test_expand_bins_by_splitting_targets_dp_compatible_count() -> None:
    bins = [[0, 1, 2], [3, 4], [5]]

    expand_bins_by_splitting(bins, target_count=4, lengths=[100, 80, 20, 50, 30, 10])

    assert len(bins) == 4
    assert sorted(index for bin_ in bins for index in bin_) == [0, 1, 2, 3, 4, 5]


def test_expand_bins_by_splitting_rejects_unsplittable_bins() -> None:
    bins = [[0], [1]]

    with pytest.raises(ValueError, match="all bins contain one sample"):
        expand_bins_by_splitting(bins, target_count=3, lengths=[10, 10])
