#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path

from lerobot.control_plane import ArtifactManifest, ReleaseRegistry
from lerobot.edge.contracts import default_edge_runtime_contract
from lerobot.edge.deployment_loop import EdgeDeploymentLoop, EdgeDeploymentLoopConfig
from lerobot.edge.mock_runtime import StaticActionRuntime, StaticActionRuntimeConfig
from lerobot.edge.model_manager import EdgeModelManager
from lerobot.edge.watchdog import WatchdogIncident
from lerobot.edge.spool import EdgeEpisodeSpool
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.robots.utils import make_robot_from_config
from tests.mocks.mock_robot import MockRobotConfig


def _write_artifact(root: Path, artifact_id: str) -> None:
    artifact_dir = root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "policy").mkdir()
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        base_checkpoint="ckpt-1",
        runtime_abi_version="v1",
        observation_schema_version="v1",
        action_schema_version="v1",
        env_processor_digest="sha256:env",
        action_processor_digest="sha256:act",
        stats_digest="sha256:stats",
        policy_digest="sha256:policy",
        compatible_robot_types=["mock_robot"],
        compatible_camera_layouts=["front_single_rgb"],
        eval_summary={"offline_score": 1.0},
        policy_config={"type": "act"},
        metadata={"policy_path": "policy", "runtime": {"device": "cpu"}},
    )
    manifest.write_json(artifact_dir / "manifest.json")


def test_deployment_loop_switches_target_at_episode_boundary(tmp_path):
    model_root = tmp_path / "models"
    registry_root = tmp_path / "registry"
    _write_artifact(model_root, "artifact-prod-001")
    _write_artifact(model_root, "artifact-prod-002")

    registry = ReleaseRegistry(registry_root)
    registry.set_channel_target(channel="prod", target_artifact_id="artifact-prod-001")
    registry.assign_device(device_id="edge-01", channel="prod")

    robot = make_robot_from_config(
        MockRobotConfig(n_motors=3, random_values=False, static_values=[1.0, 2.0, 3.0])
    )
    runtime = StaticActionRuntime(StaticActionRuntimeConfig(action_dim=3, actions_per_chunk=2, action_value=0.5))
    spool = EdgeEpisodeSpool(tmp_path / "spool")
    manager = EdgeModelManager(model_root)

    robot.connect()
    try:
        loop = EdgeDeploymentLoop(
            robot=robot,
            runtime=runtime,
            spool=spool,
            model_manager=manager,
            registry=registry,
            env_cfg=HILSerlRobotEnvConfig(),
            loop_cfg=EdgeDeploymentLoopConfig(
                processor_bundle_id="processor-v1",
                task_id="pick",
                channel="prod",
                camera_layout="front_single_rgb",
                num_episodes=2,
                fps=20,
                max_steps=1,
                use_action_processor=False,
                upload_after_run=False,
                upload_steps_per_chunk=64,
                ingestion_store_root=str(tmp_path / "ingestion"),
                device_id="edge-01",
            ),
            runtime_contract=default_edge_runtime_contract(),
            runtime_device="cpu",
        )

        def before_episode(episode_idx: int) -> None:
            if episode_idx == 1:
                registry.set_channel_target(channel="prod", target_artifact_id="artifact-prod-002")

        result = loop.run(before_episode=before_episode)
    finally:
        robot.disconnect()

    assert result.episodes_run == 2
    assert result.policy_artifact_ids == ["artifact-prod-001", "artifact-prod-002"]
    assert result.sync_actions == ["staged", "staged"]


def test_deployment_loop_rolls_back_on_model_crash(monkeypatch, tmp_path):
    model_root = tmp_path / "models"
    registry_root = tmp_path / "registry"
    _write_artifact(model_root, "artifact-prod-001")
    _write_artifact(model_root, "artifact-prod-002")

    registry = ReleaseRegistry(registry_root)
    registry.set_channel_target(channel="prod", target_artifact_id="artifact-prod-002")
    registry.assign_device(device_id="edge-01", channel="prod")

    robot = make_robot_from_config(
        MockRobotConfig(n_motors=3, random_values=False, static_values=[1.0, 2.0, 3.0])
    )
    spool = EdgeEpisodeSpool(tmp_path / "spool")
    manager = EdgeModelManager(model_root)

    from lerobot.edge.runtime import LocalPolicyRuntime

    class DummyRuntime(LocalPolicyRuntime):
        def __init__(self, name: str):
            self.name = name

    initial_runtime = DummyRuntime("active")

    def fake_sync_to_registry_target(*args, **kwargs):
        manager.state.previous_active_artifact_id = "artifact-prod-001"
        manager.state.active_artifact_id = "artifact-prod-002"
        manager._write_state()
        return type(
            "SyncResult",
            (),
            {
                "action": "activated",
                "channel": "prod",
                "target_artifact_id": "artifact-prod-002",
                "changed": True,
                "runtime": DummyRuntime("candidate"),
            },
        )()

    def fake_rollback_to_previous_active(*, runtime_overrides=None, warmup_observation=None):
        manager.state.active_artifact_id = "artifact-prod-001"
        manager.state.pending_artifact_id = None
        manager._write_state()
        return DummyRuntime("rolled-back")

    monkeypatch.setattr(manager, "sync_to_registry_target", fake_sync_to_registry_target)
    monkeypatch.setattr(manager, "rollback_to_previous_active", fake_rollback_to_previous_active)

    robot.connect()
    try:
        loop = EdgeDeploymentLoop(
            robot=robot,
            runtime=initial_runtime,
            spool=spool,
            model_manager=manager,
            registry=registry,
            env_cfg=HILSerlRobotEnvConfig(),
            loop_cfg=EdgeDeploymentLoopConfig(
                processor_bundle_id="processor-v1",
                task_id="pick",
                channel="prod",
                camera_layout="front_single_rgb",
                num_episodes=1,
                fps=20,
                max_steps=1,
                use_action_processor=False,
                upload_after_run=False,
                upload_steps_per_chunk=64,
                ingestion_store_root=str(tmp_path / "ingestion"),
                incident_store_root=str(tmp_path / "incidents"),
                device_id="edge-01",
            ),
            runtime_contract=default_edge_runtime_contract(),
            runtime_device="cpu",
        )

        def fake_run_episode(self, max_steps, fps):
            return type(
                "RunnerResult",
                (),
                {
                    "episode_id": "episode-1",
                    "step_count": 0,
                    "duration_s": 0.0,
                    "watchdog_incident": WatchdogIncident(
                        code="model_crash",
                        action="rollback_or_safe_stop",
                        message="runtime crashed",
                    ),
                },
            )()

        monkeypatch.setattr("lerobot.edge.deployment_loop.EdgeRobotRunner.run_episode", fake_run_episode)
        result = loop.run()
    finally:
        robot.disconnect()

    assert result.rollback_actions == ["rolled_back"]
    assert isinstance(loop.runtime, DummyRuntime)
    assert loop.runtime.name == "rolled-back"
    assert manager.get_active_artifact_id() == "artifact-prod-001"
    with (tmp_path / "incidents" / "episode-1.json").open("r", encoding="utf-8") as handle:
        incident = json.load(handle)
    assert incident["rollback_action"] == "rolled_back"
    assert incident["watchdog_incident"]["code"] == "model_crash"
