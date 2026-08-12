"""Token-budget packing helpers."""

from __future__ import annotations


def first_fit_pack(lengths: list[int], max_tokens_per_bin: int) -> list[list[int]]:
    """Greedily packs samples into token-budget bins.

    Processes samples in order and places each one in the first existing bin
    where its length fits without exceeding ``max_tokens_per_bin``. If no bin
    fits, it creates a new one; an oversized sample is placed alone.

    Returns indices into ``lengths`` rather than the lengths themselves.
    """

    if max_tokens_per_bin <= 0:
        raise ValueError("max_tokens_per_bin must be positive")
    if any(length < 0 for length in lengths):
        raise ValueError("sample lengths must be non-negative")

    bins: list[list[int]] = []
    bin_sums: list[int] = []
    for index, length in enumerate(lengths):
        for bin_index, bin_sum in enumerate(bin_sums):
            if bin_sum + length <= max_tokens_per_bin:
                bins[bin_index].append(index)
                bin_sums[bin_index] += length
                break
        else:
            bins.append([index])
            bin_sums.append(length)

    return bins


def split_bin_by_tokens(
    bin_indices: list[int],
    lengths: list[int],
) -> tuple[list[int], list[int]]:
    """Splits one multi-sample bin into two approximately equal-token bins."""

    if len(bin_indices) < 2:
        raise ValueError("a bin must contain at least two samples to split")
    if any(index < 0 or index >= len(lengths) for index in bin_indices):
        raise ValueError("bin contains an out-of-range sample index")

    halves: list[list[int]] = [[], []]
    half_sums = [0, 0]
    for index in sorted(bin_indices, key=lambda item: -lengths[item]):
        half = 0 if half_sums[0] <= half_sums[1] else 1
        halves[half].append(index)
        half_sums[half] += lengths[index]

    return halves[0], halves[1]


def expand_bins_by_splitting(
    bins: list[list[int]],
    target_count: int,
    lengths: list[int],
) -> None:
    """Splits the largest multi-sample bins until target_count is reached."""

    if target_count < len(bins):
        raise ValueError("target_count cannot be smaller than the current bin count")

    while len(bins) < target_count:
        candidates = [
            (sum(lengths[index] for index in bin_indices), bin_index)
            for bin_index, bin_indices in enumerate(bins)
            if len(bin_indices) > 1
        ]
        if not candidates:
            raise ValueError("cannot reach target_count: all bins contain one sample")

        _, bin_index = max(candidates)
        left, right = split_bin_by_tokens(bins[bin_index], lengths)
        bins[bin_index] = left
        bins.append(right)
