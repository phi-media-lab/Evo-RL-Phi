#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.processor import DataProcessorPipeline, EnvTransition
from lerobot.rl.gym_manipulator import make_processors
from lerobot.teleoperators.teleoperator import Teleoperator


@dataclass(frozen=True)
class EdgeProcessorBundle:
    """Runtime processors used on the edge for observation and action handling."""

    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition]
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition]

    def reset(self) -> None:
        self.env_processor.reset()
        self.action_processor.reset()


@dataclass(frozen=True)
class EdgeRobotEnvAdapter:
    """Small adapter so HIL processor builders can reuse a bare robot instance."""

    robot: Any


def make_edge_processor_bundle(
    env: Any,
    teleop_device: Teleoperator | None,
    cfg: HILSerlRobotEnvConfig,
    device: str = "cpu",
) -> EdgeProcessorBundle:
    """Build edge-safe env/action processors from the existing HIL-SERL stack."""

    processor_env = env if hasattr(env, "robot") else EdgeRobotEnvAdapter(robot=env)
    env_processor, action_processor = make_processors(
        env=processor_env,
        teleop_device=teleop_device,
        cfg=cfg,
        device=device,
    )
    return EdgeProcessorBundle(env_processor=env_processor, action_processor=action_processor)
