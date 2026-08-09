# Asuka

Asuka is my attempt to build a small, educational version of **SLIME** in order to understand distributed RL training systems for language models.

The goal is to learn SLIME’s architecture by rebuilding its core ideas in a minimal form:

```text
Ray orchestration
  → SGLang rollout workers
  → Asuka data buffer / trajectory pipeline
  → Megatron-based RL trainer
  → weight synchronization back to rollout workers
```

It is a learning project focused on understanding:

- how SLIME structures distributed RL training;
- how Ray actors and placement groups coordinate training/rollout workers;
- how SGLang is used for high-throughput rollout generation;
- how Megatron handles distributed and MoE model training;
- how rollout data becomes RL training data;
- how rewards, masks, logprobs, and model versions flow through the system;
- how weights are synchronized between trainer and rollout engines;
- how async rollout/training and staleness work.

FSDP, veRL, vLLM, and other systems are mostly studied as comparison points: useful for understanding alternative designs.

> the name is a reference to **Tanaka Asuka** from _Hibike! Euphonium_ - not the one from Evangelion
