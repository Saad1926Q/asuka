# Data-parallel scheduling

This note covers how Asuka turns `TrainData` into a plan for multiple
data-parallel GPUs.

## Purpose

Each data-parallel rank owns a model replica and processes different samples.
`build_dp_schedule()` decides:

```text
which samples form each microbatch
which microbatches each rank processes
```

It does not run the model or compute gradients.

## Flow

```text
TrainData
  -> group samples by rollout_id into global steps
  -> create microbatches
  -> align microbatch count
  -> assign complete microbatches to ranks
  -> create rank-local partitions and indices
```

`global_batch_size` counts rollout IDs per global step. Samples belonging to
the same rollout stay in the same step, even when one rollout contributes
multiple trainable samples.

## Static batching

Static mode uses `micro_batch_size`:

```text
samples:       [0, 1, 2, 3]
micro size:    2
microbatches:  [[0, 1], [2, 3]]
```

Static microbatch counts must already satisfy the DP/VPP alignment requirement;
they are not split automatically.

## Dynamic batching

Dynamic mode uses `first_fit_pack()`:

```text
sample lengths:       [3, 3, 2, 4]
token budget:         6
microbatches:         [[0, 1], [2, 3]]
```

Samples are placed in the first existing bin where they fit. If the number of
bins cannot be divided across ranks, `expand_bins_by_splitting()` splits
multi-sample bins until an aligned count is reached. It never creates fake
samples.

## Rank assignment

Default assignment is strided. For six microbatches and two ranks:

```text
rank 0 -> microbatches [0, 2, 4]
rank 1 -> microbatches [1, 3, 5]
```

With `balance_data=True`, `balance_microbatches()` uses token workload and a
multiway Karmarkar-Karp heuristic to rearrange complete microbatches into
roughly equal-workload groups.

FLOPs-aware balancing remains intentionally unsupported until model-specific
FLOPs metadata is available.

## Schedule output

`DPSchedule` stores:

- `partitions`: global sample IDs owned by each rank;
- `micro_batch_indices`: local offsets for that rank's microbatches;
- `num_microbatches`: microbatch count per rank for each global step;
- `global_batch_sizes`: rollout count for each global step.

For example:

```python
partitions = [[0, 2], [1, 3]]
micro_batch_indices = [
    [[0], [1]],
    [[0], [1]],
]
```

Rank 0 owns global samples `[0, 2]`, but its trainer sees local positions
`[0, 1]`. The local indices describe the boundaries of its two microbatches.

The scheduler validates that every sample is assigned exactly once and that all
ranks have matching microbatch counts.
