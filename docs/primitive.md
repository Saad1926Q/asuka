*primitive.md*       Asuka

                            ASUKA REFERENCE MANUAL

Core primitives                                             *asuka-primitives*

This note explains Asuka's first data contracts. These objects connect rollout
generation to training. SGLang will produce Samples, conversion code will
assemble TrainData, and Megatron will consume TrainData.

==============================================================================
Overview                                                     *primitive-overview*

Asuka follows the SLIME data path:

```text
rollout backend
  -> Sample
  -> grouped samples
  -> TrainData
  -> training backend
```

The early code does not implement rollout or training yet. It defines the
stable boundary those systems must use.

==============================================================================
Sample                                                       *primitive-sample*

A Sample is one trainable completion or chunk produced by rollout.

For one-step LLM RL:

```text
prompt:   "Solve 2 + 3"
response: "5"
Sample:  prompt + response + reward + logprobs + metadata
```

For multi-turn RL, one environment interaction may produce several Samples.
Each Sample is one trainable chunk, not necessarily the whole episode.

Important fields:

- prompt: Original task or conversation prefix.
- response: Model-generated text for this trainable chunk.
- tokens: Token ids used by the trainer.
- response_length: Number of response tokens at the end of tokens.
- reward: Processed training reward.
- raw_reward: Original verifier/environment reward.
- loss_mask: Response-token mask; 1 trains, 0 ignores.
- rollout_log_probs: Old policy logprobs from rollout.
- policy_version: Model version that generated this Sample.
- rollout_routed_experts: MoE expert ids per response token.

==============================================================================
Rollout identity                                             *primitive-rollout-id*

Use three ids for different concepts:

```text
group_id    = alternatives for the same prompt/task
rollout_id  = one environment interaction / trajectory
sample_id   = one trainable chunk
```

For GRPO with one-step prompts:

```text
prompt: "2 + 3"
group_id = 7

sample_id=0, rollout_id=100, response="5"
sample_id=1, rollout_id=101, response="6"
sample_id=2, rollout_id=102, response="The answer is 5"
sample_id=3, rollout_id=103, response="23"
```

For multi-turn agent RL:

```text
task: "fix this bug"
rollout_id = 200

sample_id=0 -> first assistant/tool action
sample_id=1 -> second assistant/tool action
sample_id=2 -> final answer
```

All chunks from the same environment interaction share rollout_id. This
prevents the trainer from over-counting one long rollout just because it was
split into several trainable pieces.

==============================================================================
SampleStatus                                                 *primitive-status*

SampleStatus records the lifecycle of one Sample.

- PENDING: Created but not finished.
- COMPLETED: Usable for training.
- TRUNCATED: Generation hit max tokens.
- ABORTED: Interrupted, usually by shutdown or weight update.
- FAILED: Recoverable rollout/reward/tool failure.

Only completed samples should normally enter training. Aborted samples may be
retried or requeued by later rollout code.

==============================================================================
TrainData                                                     *primitive-traindata*

TrainData is a batch assembled from many Sample objects.

It is column-oriented:

```text
Sample 0 tokens     -> TrainData.tokens[0]
Sample 0 reward     -> TrainData.rewards[0]
Sample 0 loss_mask  -> TrainData.loss_masks[0]

Sample 1 tokens     -> TrainData.tokens[1]
Sample 1 reward     -> TrainData.rewards[1]
Sample 1 loss_mask  -> TrainData.loss_masks[1]
```

All TrainData fields must have the same batch length. If there are 32 token
sequences, there must also be 32 rewards, 32 masks, 32 rollout ids, and so on.

This invariant is checked in `TrainData.__post_init__()`.

==============================================================================
Batch validation                                              *primitive-batch-validation*

`TrainData.__post_init__()` computes field lengths:

```python
lengths = {
    "tokens": 32,
    "rewards": 32,
    "loss_masks": 32,
}
```

Then it raises if any field has a different length.

Bad batch:

```python
tokens = [[1], [2]]
rewards = [1.0]
```

This is invalid: two samples, one reward. Failing early is better than silently
training on misaligned data.

==============================================================================
Grouped rollout output                                        *primitive-groups*

Rollout output is grouped before flattening:

```python
list[list[Sample]]
```

Outer list: prompt/task groups.
Inner list: samples or trainable chunks for that group.

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

`validate_sample_groups()` rejects malformed rollout output before training.

==============================================================================
Design rules                                                  *primitive-rules*

- Keep contracts backend-neutral.
- Let SGLang produce Samples.
- Let conversion code produce TrainData.
- Let Megatron consume TrainData.
- Preserve ids; ids are how async and multi-turn RL stay debuggable.
- Do not hide bad batches. Raise early.
