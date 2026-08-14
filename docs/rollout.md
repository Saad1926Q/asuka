# Rollout integration

This note covers Asuka's current SGLang rollout path.

## Current flow

```text
prompt text
  -> model tokenizer/chat template
  -> SGLang HTTP /generate
  -> GenerationResult
  -> Sample
  -> rewarded Sample groups
```

SGLang runs as an external server and owns model inference and GPU execution.
Asuka only implements the client and the conversion around that server.

## SGLang client

`asuka/rollout/sglang.py` contains `SGLangClient`. It sends:

```text
input_ids + sampling parameters
```

to `/generate` with `return_logprob=True`. It parses:

```text
response text
response token IDs
per-token log-probabilities
SGLang metadata
```

into `GenerationResult`.

The client is asynchronous because several generation requests should be in
flight while Asuka waits for the GPU server.

## Generation to Sample

`generation_to_sample()` converts one `GenerationResult` into one completed
`Sample`:

```text
prompt tokens + response tokens -> Sample.tokens
response token count            -> Sample.response_length
response positions              -> Sample.loss_mask
SGLang logprobs                 -> Sample.rollout_log_probs
```

The prompt tokens provide context for future training. The response tokens are
marked trainable by the loss mask.

## Grouped generation

`generate_group()` handles one prompt and several alternatives concurrently.
All alternatives share a `group_id` but have distinct `rollout_id` and
`sample_id` values:

```text
one prompt
  -> response A: group=10, rollout=100, sample=200
  -> response B: group=10, rollout=101, sample=201
```

The reward function receives each completed Sample. It may be synchronous for a
local verifier or asynchronous for a remote reward service.

`generate_batch()` repeats this for several prompts, assigns non-overlapping ID
ranges, and preserves prompt/alternative order.

## Current smoke test

`scripts/smoke/run_rollout_pipeline.py` uses:

```text
model: LiquidAI/LFM2.5-350M
prompts: fixed arithmetic prompt dictionary
alternatives: two per prompt
reward: exact-match answer checking
```

The server is started separately:

```bash
python -m sglang.launch_server \
  --model-path LiquidAI/LFM2.5-350M \
  --host 127.0.0.1 \
  --port 30000
```

Then run:

```bash
uv run scripts/smoke/run_rollout_pipeline.py
```

The next rollout-layer work is server lifecycle management and eventually Ray
placement. Those are intentionally not part of the current HTTP adapter.
