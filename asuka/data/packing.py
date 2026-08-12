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
