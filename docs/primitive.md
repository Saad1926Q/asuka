# Asuka data primitives

This document describes the backend-neutral data objects shared between rollout
generation and training.

## `Sample`

A `Sample` is one trainable completion or chunk produced by a rollout.

For a one-step task:

```text
prompt:   "Solve 2 + 3"
response: "5"
Sample:   prompt + response + reward + log-probabilities
```

For multi-turn RL, one trajectory may produce several Samples. Each Sample is
one trainable chunk, not necessarily the entire trajectory.

Important fields:

- `prompt`: original task or conversation prefix;
- `response`: generated text for this trainable chunk;
- `tokens`: prompt and response token IDs used by the trainer;
- `response_length`: number of response tokens;
- `reward`: processed reward used for training;
- `raw_reward`: original verifier or environment reward;
- `loss_mask`: response-token mask, where `1` trains and `0` ignores;
- `rollout_log_probs`: old policy log-probabilities from generation;
- `policy_version`: policy version that generated the Sample;
- `rollout_routed_experts`: routed MoE experts for response tokens.

## Sample identities

The three IDs represent different things:

```text
group_id    = alternatives for the same prompt/task
rollout_id  = one environment interaction or trajectory
sample_id   = one trainable chunk
```

For GRPO with one-step prompts:

```text
prompt: "2 + 3"
group_id = 7

sample_id=0, rollout_id=100, response="5"
sample_id=1, rollout_id=101, response="6"
sample_id=2, rollout_id=102, response="The answer is 5"
```

For a multi-turn trajectory:

```text
rollout_id = 200

sample_id=0 -> first assistant/tool action
sample_id=1 -> second assistant/tool action
sample_id=2 -> final answer
```

All chunks from the same trajectory share `rollout_id`.

## `SampleStatus`

`SampleStatus` records the lifecycle of a Sample:

- `PENDING`: created but not finished;
- `COMPLETED`: usable for training;
- `TRUNCATED`: generation reached its token limit;
- `ABORTED`: interrupted by shutdown or another control event;
- `FAILED`: recoverable generation, reward, or tool failure.

Only completed Samples should normally enter training.

## `TrainData`

`TrainData` is a batch assembled from Samples. It is column-oriented:

```text
Sample 0 tokens     -> TrainData.tokens[0]
Sample 0 reward     -> TrainData.rewards[0]
Sample 0 loss_mask  -> TrainData.loss_masks[0]

Sample 1 tokens     -> TrainData.tokens[1]
Sample 1 reward     -> TrainData.rewards[1]
Sample 1 loss_mask  -> TrainData.loss_masks[1]
```

Every field must have the same batch length. If there are 32 token sequences,
there must also be 32 rewards, masks, rollout IDs, and so on.

This invariant is checked in `TrainData.__post_init__()`.

## Grouped rollout output

Rollout output is grouped before flattening:

```python
list[list[Sample]]
```

The outer list contains prompt/task groups. Each inner list contains the
alternatives or trainable chunks belonging to one group.

Example:

```python
[
    [sample_a0, sample_a1],
    [sample_b0, sample_b1],
]
```

`flatten_sample_groups()` converts this to:

```python
[sample_a0, sample_a1, sample_b0, sample_b1]
```

`validate_sample_groups()` checks that the structure and group identities are
valid before conversion to `TrainData`.

## Reward normalization

`normalize_group_rewards()` compares alternatives within each `group_id`.

For example:

```text
raw rewards:  [1.0, 0.0]
mean:         0.5
normalized:   [0.5, -0.5]
```

This gives the trainer a relative signal: the first alternative performed
better than the second for the same prompt.
