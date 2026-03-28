#!/usr/bin/env python

from __future__ import annotations

import math
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .runtime_protocol import EdgePolicyRuntime


@dataclass
class LocalPolicyRuntimeConfig:
    """Configuration for a single-machine eager inference runtime."""

    policy_type: str
    pretrained_name_or_path: str
    device: str = "cpu"
    task: str = ""
    robot_type: str = ""
    actions_per_chunk: int = 8
    chunk_size_threshold: float = 0.5
    use_amp: bool = False
    rename_map: dict[str, str] = field(default_factory=dict)
    preprocessor_config_filename: str | None = None
    postprocessor_config_filename: str | None = None

    def __post_init__(self) -> None:
        if not self.policy_type:
            raise ValueError("policy_type cannot be empty.")
        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty.")
        if self.actions_per_chunk <= 0:
            raise ValueError("actions_per_chunk must be positive.")
        if not 0 <= self.chunk_size_threshold <= 1:
            raise ValueError("chunk_size_threshold must be between 0 and 1.")


class LocalPolicyRuntime:
    """Local eager policy runtime with a small action queue for edge control loops."""

    def __init__(self, config: LocalPolicyRuntimeConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.policy = self._load_policy()
        self.preprocessor, self.postprocessor = self._load_processors()
        self.action_queue: deque[torch.Tensor] = deque()
        self.last_chunk: torch.Tensor | None = None

    @property
    def queue_size(self) -> int:
        return len(self.action_queue)

    @property
    def chunk_refill_watermark(self) -> int:
        return max(1, math.ceil(self.config.actions_per_chunk * self.config.chunk_size_threshold))

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.action_queue.clear()
        self.last_chunk = None

    def warmup(self, observation: dict[str, Any]) -> torch.Tensor:
        """Run one eager forward pass and prime the action queue."""

        self.action_queue.clear()
        return self.refill_action_queue(observation)

    def maybe_refill_action_queue(self, observation: dict[str, Any]) -> torch.Tensor | None:
        if self.queue_size > self.chunk_refill_watermark:
            return None
        return self.refill_action_queue(observation)

    def refill_action_queue(self, observation: dict[str, Any]) -> torch.Tensor:
        chunk = self.predict_action_chunk(observation)
        self.last_chunk = chunk
        self.action_queue.extend(chunk.unbind(0))
        return chunk

    def pop_next_action(self, observation: dict[str, Any]) -> torch.Tensor:
        self.maybe_refill_action_queue(observation)
        if not self.action_queue:
            self.refill_action_queue(observation)
        return self.action_queue.popleft()

    def predict_action_chunk(self, observation: dict[str, Any]) -> torch.Tensor:
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and self.config.use_amp
            else nullcontext(),
        ):
            prepared_observation = self._prepare_observation(observation)
            processed_observation = self.preprocessor(prepared_observation)
            raw_chunk = self.policy.predict_action_chunk(processed_observation)
            processed_chunk = self._postprocess_action_chunk(raw_chunk)

        return processed_chunk

    def _load_policy(self) -> PreTrainedPolicy:
        policy_class = get_policy_class(self.config.policy_type)
        policy = policy_class.from_pretrained(self.config.pretrained_name_or_path)
        policy.to(self.device)
        policy.eval()
        return policy

    def _load_processors(
        self,
    ) -> tuple[
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        PolicyProcessorPipeline[PolicyAction, PolicyAction],
    ]:
        preprocessor_overrides: dict[str, dict[str, Any]] = {
            "device_processor": {"device": self.config.device},
        }
        if self.config.rename_map:
            preprocessor_overrides["rename_observations_processor"] = {"rename_map": self.config.rename_map}

        kwargs: dict[str, Any] = {
            "pretrained_path": self.config.pretrained_name_or_path,
            "preprocessor_overrides": preprocessor_overrides,
            "postprocessor_overrides": {"device_processor": {"device": self.config.device}},
        }
        if self.config.preprocessor_config_filename is not None:
            kwargs["preprocessor_config_filename"] = self.config.preprocessor_config_filename
        if self.config.postprocessor_config_filename is not None:
            kwargs["postprocessor_config_filename"] = self.config.postprocessor_config_filename

        return make_pre_post_processors(self.policy.config, **kwargs)

    def _prepare_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        prepared = deepcopy(observation)
        for name, value in list(prepared.items()):
            if isinstance(value, torch.Tensor):
                tensor = value.clone()
            else:
                tensor = torch.from_numpy(np.asarray(value))

            if "image" in name and tensor.ndim == 3:
                tensor = tensor.to(dtype=torch.float32) / 255.0
                tensor = tensor.permute(2, 0, 1).contiguous()

            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim >= 1 and tensor.shape[0] != 1:
                tensor = tensor.unsqueeze(0)

            prepared[name] = tensor.to(self.device)

        prepared["task"] = self.config.task
        prepared["robot_type"] = self.config.robot_type
        return prepared

    def _postprocess_action_chunk(self, action_chunk: torch.Tensor) -> torch.Tensor:
        if action_chunk.ndim == 1:
            action_chunk = action_chunk.unsqueeze(0).unsqueeze(0)
        elif action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(0)

        action_chunk = action_chunk[:, : self.config.actions_per_chunk, :]
        processed_actions: list[torch.Tensor] = []
        for action_idx in range(action_chunk.shape[1]):
            action = action_chunk[:, action_idx, :]
            processed_action = self.postprocessor(action)
            processed_actions.append(processed_action.detach().cpu())
        return torch.stack(processed_actions, dim=1).squeeze(0)


class LocalPolicyRuntimeFactory:
    runtime_config_type = LocalPolicyRuntimeConfig

    def build_runtime(
        self,
        runtime_config: LocalPolicyRuntimeConfig,
        *,
        warmup_observation: dict[str, Any] | None = None,
    ) -> EdgePolicyRuntime:
        runtime = LocalPolicyRuntime(runtime_config)
        if warmup_observation is not None:
            runtime.warmup(warmup_observation)
        return runtime

    def supports_runtime(self, runtime: EdgePolicyRuntime) -> bool:
        return isinstance(runtime, LocalPolicyRuntime)
