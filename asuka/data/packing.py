"""Token-budget packing and workload-balancing helpers."""

from __future__ import annotations


def first_fit_pack(lengths: list[int], max_tokens_per_bin: int) -> list[list[int]]:
    """Greedily packs samples into token-budget bins.

    Processes samples in order and places each one in the first existing bin
    where its length fits without exceeding ``max_tokens_per_bin``. If no bin
    fits, it creates a new one; an oversized sample is placed alone.

    Args:
        lengths: Token length of each sample, indexed by sample ID.
        max_tokens_per_bin: Maximum total tokens allowed in a normal bin.

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
    """Splits one multi-sample bin into two approximately equal-token bins.

    Args:
        bin_indices: Sample IDs contained in the bin to split.
        lengths: Token length of every sample, indexed by sample ID.
    """

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
    """Splits the largest multi-sample bins until ``target_count`` is reached.

    Args:
        bins: Mutable bins containing sample IDs; expanded in place.
        target_count: Desired total number of bins.
        lengths: Token length of every sample, indexed by sample ID.
    """

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


def _group_workload(group: list[int], workloads: list[int]) -> int:
    """Returns the total workload of a microbatch-ID group.

    Args:
        group: Microbatch IDs belonging to one group.
        workloads: Workload of each microbatch, indexed by microbatch ID.
    """

    return sum(workloads[index] for index in group)


def _sort_workload_groups(
    groups: list[list[int]],
    workloads: list[int],
) -> list[list[int]]:
    """Sorts groups from largest to smallest workload.

    Args:
        groups: Groups of microbatch IDs to sort.
        workloads: Workload of each microbatch, indexed by microbatch ID.
    """

    return sorted(
        groups, key=lambda group: (_group_workload(group, workloads), group), reverse=True
    )


def _state_key(state: list[list[int]], workloads: list[int]) -> tuple[int, int]:
    """Returns a state priority key based on workload spread and maximum.

    Args:
        state: Groups of microbatch IDs representing one candidate state.
        workloads: Workload of each microbatch, indexed by microbatch ID.
    """

    sums = [_group_workload(group, workloads) for group in state]
    # The spread measures imbalance; max(sums) breaks ties between equal spreads.
    return (max(sums) - min(sums), max(sums))


def balance_microbatches(
    microbatches: list[list[int]],
    lengths: list[int],
    dp_size: int,
) -> list[list[int]]:
    """Partitions microbatch IDs into equal-sized, workload-balanced groups.

    Uses the multiway Karmarkar-Karp (largest differencing) heuristic. It first
    computes each microbatch's workload, creates equal-sized partial states,
    then repeatedly merges states by pairing large groups with small groups.
    The final state contains the balanced groups of microbatch IDs.

    Args:
        microbatches: Microbatches whose inner lists contain sample IDs.
        lengths: Token length of every sample, indexed by sample ID.
        dp_size: Number of output groups to create.
    """

    if dp_size <= 0:
        raise ValueError("dp_size must be positive")
    if not microbatches or len(microbatches) % dp_size != 0:
        raise ValueError("microbatch count must be a non-zero multiple of dp_size")

    workloads = [sum(lengths[index] for index in microbatch) for microbatch in microbatches]

    ordered = sorted(range(len(microbatches)), key=lambda index: (workloads[index], index))

    states: list[list[list[int]]] = []

    for start in range(0, len(ordered), dp_size):
        state = [[ordered[start + offset]] for offset in range(dp_size)]

        states.append(_sort_workload_groups(state, workloads))

    # Keep merging partial partitions until one complete partition remains.
    while len(states) > 1:
        first_index = max(
            range(len(states)), key=lambda index: _state_key(states[index], workloads)
        )
        first = states.pop(first_index)
        second_index = max(
            range(len(states)), key=lambda index: _state_key(states[index], workloads)
        )
        second = states.pop(second_index)
        # Pair large groups from one state with small groups from the other.
        merged = [first[index] + second[dp_size - 1 - index] for index in range(dp_size)]

        states.append(_sort_workload_groups(merged, workloads))

    return [sorted(group) for group in states[0]]
