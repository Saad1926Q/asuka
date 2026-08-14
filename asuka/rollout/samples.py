"""Convert backend generation results into Asuka training samples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from asuka.data.contracts import Sample, SampleStatus
from asuka.rollout.sglang import GenerationResult


def generation_to_sample(
    prompt: str,
    prompt_tokens: list[int],
    result: GenerationResult,
    *,
    group_id: int,
    rollout_id: int,
    sample_id: int,
    policy_version: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> Sample:
    """Builds a completed training sample from one generated response.

    Args:
        prompt: Original prompt text sent to the rollout model.
        prompt_tokens: Token IDs representing the prompt.
        result: Generated response text, tokens, log-probs, and metadata.
        group_id: ID shared by alternatives sampled from the same prompt.
        rollout_id: ID of this generated trajectory.
        sample_id: Unique ID of this trainable sample.
        policy_version: Version of the policy that generated the response.
        metadata: Optional generation metadata copied onto the sample.
    """

    if len(result.token_ids) != len(result.log_probs):
        raise ValueError(
            "generated token and log-probability counts must match: "
            f"{len(result.token_ids)} != {len(result.log_probs)}"
        )

    response_tokens = list(result.token_ids)
    response_length = len(response_tokens)
    return Sample(
        prompt=prompt,
        response=result.text,
        tokens=[*prompt_tokens, *response_tokens],
        response_length=response_length,
        loss_mask=[1] * response_length,
        rollout_log_probs=list(result.log_probs),
        rollout_id=rollout_id,
        group_id=group_id,
        sample_id=sample_id,
        policy_version=policy_version,
        status=SampleStatus.COMPLETED,
        metadata=dict(metadata or {}) | dict(result.meta_info),
    )
