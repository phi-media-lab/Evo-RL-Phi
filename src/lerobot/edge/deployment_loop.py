#!/usr/bin/env python

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from lerobot.cloud.ingestion import FilesystemEpisodeIngestionStore, HTTPEpisodeIngestionClient
from lerobot.control_plane.registry import ReleaseRegistry
from lerobot.edge.contracts import EdgeRuntimeContract
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.robots.robot import Robot
from lerobot.teleoperators.teleoperator import Teleoperator

from .incidents import EdgeIncidentRecord, FilesystemIncidentSink
from .model_manager import EdgeCompatibilityContext, EdgeDeploymentSyncResult, EdgeModelManager
from .processing import make_edge_processor_bundle
from .runner import EdgeRobotRunner
from .runtime import LocalPolicyRuntime
from .spool import EdgeEpisodeSpool
from .uploader import EdgeEpisodeUploader, EdgeUploaderConfig


@dataclass(frozen=True)
class EdgeDeploymentLoopConfig:
    processor_bundle_id: str
    task_id: str
    channel: str
    camera_layout: str
    num_episodes: int
    fps: int
    max_steps: int
    use_action_processor: bool
    upload_after_run: bool
    upload_steps_per_chunk: int
    ingestion_store_root: str
    ingestion_base_url: str | None = None
    incident_store_root: str | None = None
    device_id: str | None = None


@dataclass(frozen=True)
class EdgeDeploymentLoopResult:
    episode_id: str
    step_count: int
    uploaded: bool
    episodes_run: int
    episode_ids: list[str] = field(default_factory=list)
    policy_artifact_ids: list[str] = field(default_factory=list)
    sync_actions: list[str] = field(default_factory=list)
    rollback_actions: list[str] = field(default_factory=list)
    upload_receipt: dict[str, Any] | None = None


class EdgeDeploymentLoop:
    """Runs edge episodes while syncing local deployment targets on episode boundaries."""

    def __init__(
        self,
        *,
        robot: Robot,
        runtime: Any,
        spool: EdgeEpisodeSpool,
        model_manager: EdgeModelManager,
        registry: ReleaseRegistry | None,
        env_cfg: HILSerlRobotEnvConfig,
        loop_cfg: EdgeDeploymentLoopConfig,
        runtime_contract: EdgeRuntimeContract,
        runtime_device: str,
        teleop: Teleoperator | None = None,
    ):
        self.robot = robot
        self.runtime = runtime
        self.spool = spool
        self.model_manager = model_manager
        self.registry = registry
        self.env_cfg = env_cfg
        self.loop_cfg = loop_cfg
        self.runtime_contract = runtime_contract
        self.runtime_device = runtime_device
        self.teleop = teleop
        self.incident_sink = (
            FilesystemIncidentSink(self.loop_cfg.incident_store_root)
            if self.loop_cfg.incident_store_root is not None
            else None
        )
        self.compatibility = EdgeCompatibilityContext(
            robot_type=self.robot.name,
            camera_layout=self.loop_cfg.camera_layout,
            runtime_abi_version=self.runtime_contract.runtime_abi_version,
            observation_schema_version=self.runtime_contract.observation_schema_version,
            action_schema_version=self.runtime_contract.action_schema_version,
        )

    def run(
        self,
        *,
        before_episode: Callable[[int], None] | None = None,
        fallback_policy_artifact_id: str = "local-dev-artifact",
        artifact_loader: Callable[[], Any | None] | None = None,
    ) -> EdgeDeploymentLoopResult:
        action_processor = self._build_action_processor()
        episode_ids: list[str] = []
        policy_artifact_ids: list[str] = []
        sync_actions: list[str] = []
        rollback_actions: list[str] = []
        upload_receipt = None
        result = None

        for episode_idx in range(self.loop_cfg.num_episodes):
            if before_episode is not None:
                before_episode(episode_idx)

            policy_artifact_id = fallback_policy_artifact_id
            sync_result = self._sync_runtime_to_target()
            if sync_result is not None:
                policy_artifact_id = sync_result.target_artifact_id
                sync_actions.append(sync_result.action)
                if sync_result.runtime is not None:
                    self.runtime = sync_result.runtime
            else:
                artifact = artifact_loader() if artifact_loader is not None else None
                if artifact is not None:
                    self.model_manager.stage_artifact(artifact, self.compatibility)
                    policy_artifact_id = artifact.manifest.artifact_id
                    if isinstance(self.runtime, LocalPolicyRuntime):
                        self.runtime = self.model_manager.activate_artifact(
                            artifact,
                            runtime_overrides={
                                "device": self.runtime_device,
                                "task": self.loop_cfg.task_id,
                                "robot_type": self.robot.name,
                            },
                        )
                sync_actions.append("static")

            runner = EdgeRobotRunner(
                robot=self.robot,
                runtime=self.runtime,
                spool=self.spool,
                policy_artifact_id=policy_artifact_id,
                processor_bundle_id=self.loop_cfg.processor_bundle_id,
                task_id=self.loop_cfg.task_id,
                channel=self.loop_cfg.channel,
                action_processor=action_processor,
                runtime_contract=self.runtime_contract,
            )
            result = runner.run_episode(max_steps=self.loop_cfg.max_steps, fps=self.loop_cfg.fps)
            episode_ids.append(result.episode_id)
            policy_artifact_ids.append(policy_artifact_id)
            rollback_action = self._maybe_handle_watchdog_incident(result)
            rollback_actions.append(rollback_action)
            self._record_incident(
                episode_id=result.episode_id,
                policy_artifact_id=policy_artifact_id,
                sync_action=sync_actions[-1],
                rollback_action=rollback_action,
                watchdog_incident=result.watchdog_incident,
            )

            if self.loop_cfg.upload_after_run:
                uploader = EdgeEpisodeUploader(
                    spool=self.spool,
                    sink=self._make_ingestion_sink(),
                    config=EdgeUploaderConfig(steps_per_chunk=self.loop_cfg.upload_steps_per_chunk),
                )
                upload_receipt = uploader.upload_episode(result.episode_id)

        if result is None:
            raise ValueError("No episode was executed.")

        return EdgeDeploymentLoopResult(
            episode_id=result.episode_id,
            step_count=result.step_count,
            uploaded=upload_receipt is not None,
            episodes_run=self.loop_cfg.num_episodes,
            episode_ids=episode_ids,
            policy_artifact_ids=policy_artifact_ids,
            sync_actions=sync_actions,
            rollback_actions=rollback_actions,
            upload_receipt=upload_receipt,
        )

    def _make_ingestion_sink(self) -> Any:
        if self.loop_cfg.ingestion_base_url is not None:
            return HTTPEpisodeIngestionClient(self.loop_cfg.ingestion_base_url)
        return FilesystemEpisodeIngestionStore(Path(self.loop_cfg.ingestion_store_root))

    def _build_action_processor(self) -> Any | None:
        if not self.loop_cfg.use_action_processor:
            return None
        if self.teleop is None:
            logging.info("Skipping action processor because no teleoperator is configured.")
            return None
        bundle = make_edge_processor_bundle(
            env=self.robot,
            teleop_device=self.teleop,
            cfg=self.env_cfg,
            device=self.runtime_device,
        )
        return bundle.action_processor

    def _sync_runtime_to_target(self) -> EdgeDeploymentSyncResult | None:
        if self.loop_cfg.device_id is None or self.registry is None:
            return None
        return self.model_manager.sync_to_registry_target(
            self.registry,
            device_id=self.loop_cfg.device_id,
            context=self.compatibility,
            fallback_channel=self.loop_cfg.channel,
            activate=isinstance(self.runtime, LocalPolicyRuntime),
            runtime_overrides={
                "device": self.runtime_device,
                "task": self.loop_cfg.task_id,
                "robot_type": self.robot.name,
            },
        )

    def _maybe_handle_watchdog_incident(self, result: Any) -> str:
        incident = result.watchdog_incident
        if incident is None:
            return "none"
        if incident.code != "model_crash":
            return "ignored"
        if incident.action != "rollback_or_safe_stop":
            return "ignored"
        if not isinstance(self.runtime, LocalPolicyRuntime):
            return "safe_stop"

        try:
            self.runtime = self.model_manager.rollback_to_previous_active(
                runtime_overrides={
                    "device": self.runtime_device,
                    "task": self.loop_cfg.task_id,
                    "robot_type": self.robot.name,
                }
            )
        except ValueError:
            return "safe_stop"
        return "rolled_back"

    def _record_incident(
        self,
        *,
        episode_id: str,
        policy_artifact_id: str,
        sync_action: str,
        rollback_action: str,
        watchdog_incident: Any,
    ) -> None:
        if self.incident_sink is None:
            return
        incident = EdgeIncidentRecord.from_watchdog(
            episode_id=episode_id,
            device_id=self.loop_cfg.device_id,
            channel=self.loop_cfg.channel,
            policy_artifact_id=policy_artifact_id,
            sync_action=sync_action,
            rollback_action=rollback_action,
            watchdog_incident=watchdog_incident,
        )
        self.incident_sink.record(incident)
