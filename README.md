# asuka (アスカ) /ᐠ - ˕ -マ Ⳋ

the main idea in mind for this project is to make a minimal distributed async RL training system for language models — in an attempt to learn how some of the frontier RL libraries work and what primitives they use.

## the plan is to learn and read about how the following work:

- **ray** - distributed orchestration, actors, queues, gpu scheduling
- **nccl** - fast weight sync between trainer and inference servers
- **grpo** - group-relative policy optimization, kl penalties, reward hacking
- **vllm** - running inference servers programmatically, weight update api
- **fsdp** - sharding models across gpus
- **async patterns** - producer/consumer, backpressure, staleness handling

> the name is a reference to **Tanaka Asuka** from _Hibike! Euphonium_ - not the one from Evangelion
