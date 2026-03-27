#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StaticActionRuntimeConfig:
    """Deterministic runtime used for dry-runs and integration testing."""

    action_dim: int
    actions_per_chunk: int = 8
    action_value: float = 0.0

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if self.actions_per_chunk <= 0:
            raise ValueError("actions_per_chunk must be positive.")


class StaticActionRuntime:
    """Small runtime with the same surface as LocalPolicyRuntime for dry-runs."""

    def __init__(self, config: StaticActionRuntimeConfig):
        self.config = config
        self.action_queue: list[torch.Tensor] = []

    @property
    def queue_size(self) -> int:
        return len(self.action_queue)

    def reset(self) -> None:
        self.action_queue.clear()

    def warmup(self, observation: dict | None = None) -> torch.Tensor:
        return self.refill_action_queue(observation or {})

    def maybe_refill_action_queue(self, observation: dict | None = None) -> torch.Tensor | None:
        if self.action_queue:
            return None
        return self.refill_action_queue(observation or {})

    def refill_action_queue(self, observation: dict | None = None) -> torch.Tensor:
        del observation
        chunk = torch.full(
            (self.config.actions_per_chunk, self.config.action_dim),
            fill_value=self.config.action_value,
            dtype=torch.float32,
        )
        self.action_queue.extend(list(chunk.unbind(0)))
        return chunk

    def pop_next_action(self, observation: dict | None = None) -> torch.Tensor:
        self.maybe_refill_action_queue(observation or {})
        if not self.action_queue:
            self.refill_action_queue(observation or {})
        return self.action_queue.pop(0)
