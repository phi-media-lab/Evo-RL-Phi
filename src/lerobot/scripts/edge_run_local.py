#!/usr/bin/env python

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any, Callable

import draccus

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.control_plane.registry import ReleaseRegistry
from lerobot.edge.contracts import EdgeRuntimeContract, default_edge_runtime_contract
from lerobot.edge.deployment_loop import EdgeDeploymentLoop, EdgeDeploymentLoopConfig
from lerobot.edge.model_manager import EdgeModelManager
from lerobot.edge.mock_runtime import StaticActionRuntime, StaticActionRuntimeConfig
from lerobot.edge.runtime import LocalPolicyRuntime, LocalPolicyRuntimeConfig
from lerobot.edge.spool import EdgeEpisodeSpool
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.utils import init_logging


@dataclass
class EdgeRunLocalConfig:
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})
    runtime: LocalPolicyRuntimeConfig = field(metadata={"help": "Local policy runtime configuration"})
    env: HILSerlRobotEnvConfig = field(default_factory=HILSerlRobotEnvConfig)
    teleop: TeleoperatorConfig | None = None
    spool_root: str = ".edge_spool"
    model_store_root: str = ".edge_models"
    registry_root: str | None = None
    device_id: str | None = None
    artifact_dir: str | None = None
    camera_layout: str = "front_single_rgb"
    policy_artifact_id: str = "local-dev-artifact"
    processor_bundle_id: str = "edge-processor-v1"
    channel: str = "dev"
    task_id: str = "default-task"
    num_episodes: int = 1
    fps: int = 20
    max_steps: int = 100
    use_action_processor: bool = True
    dry_run: bool = False
    dry_run_action_value: float = 0.0
    upload_after_run: bool = False
    ingestion_store_root: str = ".edge_ingestion"
    incident_store_root: str | None = ".edge_incidents"
    upload_steps_per_chunk: int = 64
    runtime_contract: EdgeRuntimeContract = field(default_factory=default_edge_runtime_contract)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.num_episodes <= 0:
            raise ValueError("num_episodes must be positive.")
        if self.upload_steps_per_chunk <= 0:
            raise ValueError("upload_steps_per_chunk must be positive.")
        if self.device_id is not None and not self.device_id:
            raise ValueError("device_id cannot be empty.")
        if self.registry_root is not None and not self.registry_root:
            raise ValueError("registry_root cannot be empty.")


@dataclass(frozen=True)
class EdgeRunLocalResult:
    episode_id: str
    step_count: int
    uploaded: bool
    episodes_run: int = 1
    episode_ids: list[str] = field(default_factory=list)
    policy_artifact_ids: list[str] = field(default_factory=list)
    sync_actions: list[str] = field(default_factory=list)
    rollback_actions: list[str] = field(default_factory=list)
    upload_receipt: dict[str, Any] | None = None


def run_edge_local(
    cfg: EdgeRunLocalConfig,
    *,
    before_episode: Callable[[int], None] | None = None,
) -> EdgeRunLocalResult:
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    spool = EdgeEpisodeSpool(Path(cfg.spool_root))
    model_manager = EdgeModelManager(Path(cfg.model_store_root))
    runtime_device = getattr(cfg.runtime, "device", "cpu")

    robot.connect()
    if teleop is not None:
        teleop.connect()

    try:
        if cfg.dry_run:
            runtime = StaticActionRuntime(
                StaticActionRuntimeConfig(
                    action_dim=len(robot.action_features),
                    actions_per_chunk=cfg.runtime.actions_per_chunk,
                    action_value=cfg.dry_run_action_value,
                )
            )
        else:
            runtime = LocalPolicyRuntime(cfg.runtime)

        deployment_loop = EdgeDeploymentLoop(
            robot=robot,
            runtime=runtime,
            spool=spool,
            model_manager=model_manager,
            registry=ReleaseRegistry(Path(cfg.registry_root)) if cfg.registry_root is not None else None,
            env_cfg=cfg.env,
            loop_cfg=EdgeDeploymentLoopConfig(
                processor_bundle_id=cfg.processor_bundle_id,
                task_id=cfg.task_id,
                channel=cfg.channel,
                camera_layout=cfg.camera_layout,
                num_episodes=cfg.num_episodes,
                fps=cfg.fps,
                max_steps=cfg.max_steps,
                use_action_processor=cfg.use_action_processor,
                upload_after_run=cfg.upload_after_run,
                upload_steps_per_chunk=cfg.upload_steps_per_chunk,
                ingestion_store_root=cfg.ingestion_store_root,
                incident_store_root=cfg.incident_store_root,
                device_id=cfg.device_id,
            ),
            runtime_contract=cfg.runtime_contract,
            runtime_device=runtime_device,
            teleop=teleop,
        )
        result = deployment_loop.run(
            before_episode=before_episode,
            fallback_policy_artifact_id=cfg.policy_artifact_id,
            artifact_loader=(lambda: model_manager.load_artifact(cfg.artifact_dir)) if cfg.artifact_dir else None,
        )
        return EdgeRunLocalResult(**result.__dict__)
    finally:
        if teleop is not None:
            teleop.disconnect()
        robot.disconnect()


@draccus.wrap()
def main(cfg: EdgeRunLocalConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    result = run_edge_local(cfg)
    logging.info("Edge run completed: %s", result)


if __name__ == "__main__":
    main()
