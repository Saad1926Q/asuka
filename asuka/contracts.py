"""Core data contracts for Asuka's rollout-to-training boundary.

Rollout backends produce Sample objects.
Conversion code will assemble them into TrainData.
Training backends consume TrainData.
These types intentionally stay backend-neutral.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar


class SampleStatus(StrEnum):
    """Lifecycle state for one rollout sample."""

    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass(slots=True)
class Sample:
    """One rollout completion and its training metadata."""

    prompt: str
    response: str
    tokens: list[int]
    response_length: int

    reward: float | None = None
    raw_reward: float | None = None
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None

    rollout_id: int | None = None
    group_id: int | None = None
    sample_id: int | None = None
    policy_version: int = 0

    status: SampleStatus = SampleStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    rollout_routed_experts: list[list[int]] | None = None


@dataclass(slots=True)
class TrainData:
    """Batch format consumed by training backends."""

    tokens: list[list[int]]
    response_lengths: list[int]
    rewards: list[float]
    raw_rewards: list[float]
    loss_masks: list[list[int]]
    rollout_ids: list[int]
    group_ids: list[int]
    sample_ids: list[int]
    policy_versions: list[int]
    rollout_log_probs: list[list[float] | None]
    rollout_routed_experts: list[list[list[int]] | None]
    metadata: list[dict[str, Any]]

    _BATCH_FIELDS: ClassVar[tuple[str, ...]] = (
        "tokens",
        "response_lengths",
        "rewards",
        "raw_rewards",
        "loss_masks",
        "rollout_ids",
        "group_ids",
        "sample_ids",
        "policy_versions",
        "rollout_log_probs",
        "rollout_routed_experts",
        "metadata",
    )

    def __post_init__(self) -> None:
        lengths = {name: len(getattr(self, name)) for name in self._BATCH_FIELDS}
        expected = next(iter(lengths.values()))
        mismatched = {name: length for name, length in lengths.items() if length != expected}
        if mismatched:
            raise ValueError(
                "TrainData fields must have the same batch length; "
                f"expected {expected}, mismatched={mismatched}, lengths={lengths}"
            )

    @property
    def batch_size(self) -> int:
        return len(self.tokens)
