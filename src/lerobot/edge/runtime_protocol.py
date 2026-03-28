#!/usr/bin/env python

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class EdgePolicyRuntime(Protocol):
    @property
    def queue_size(self) -> int: ...

    def reset(self) -> None: ...

    def warmup(self, observation: dict[str, Any] | None = None) -> torch.Tensor: ...

    def maybe_refill_action_queue(self, observation: dict[str, Any] | None = None) -> torch.Tensor | None: ...

    def refill_action_queue(self, observation: dict[str, Any] | None = None) -> torch.Tensor: ...

    def pop_next_action(self, observation: dict[str, Any] | None = None) -> torch.Tensor: ...


class EdgeRuntimeFactory(Protocol):
    def build_runtime(
        self,
        runtime_config: Any,
        *,
        warmup_observation: dict[str, Any] | None = None,
    ) -> EdgePolicyRuntime: ...

    def supports_runtime(self, runtime: EdgePolicyRuntime) -> bool: ...
