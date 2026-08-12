import pytest

from asuka.data.packing import first_fit_pack


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
