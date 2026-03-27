#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path

from lerobot.control_plane import ArtifactManifest, ReleaseRegistry
from lerobot.edge.mock_runtime import StaticActionRuntimeConfig
from lerobot.scripts.edge_run_local import EdgeRunLocalConfig, run_edge_local
from tests.edge.test_http_ingestion import _start_ingestion_server
from tests.mocks.mock_robot import MockRobotConfig


def test_run_edge_local_dry_run_with_upload(tmp_path):
    cfg = EdgeRunLocalConfig(
        robot=MockRobotConfig(
            n_motors=3,
            random_values=False,
            static_values=[1.0, 2.0, 3.0],
        ),
        runtime=StaticActionRuntimeConfig(
            action_dim=3,
            actions_per_chunk=2,
            action_value=0.75,
        ),
        spool_root=str(tmp_path / "spool"),
        model_store_root=str(tmp_path / "models"),
        ingestion_store_root=str(tmp_path / "ingestion"),
        max_steps=3,
        fps=20,
        dry_run=True,
        upload_after_run=True,
        upload_steps_per_chunk=2,
    )

    result = run_edge_local(cfg)

    assert result.step_count == 3
    assert result.uploaded is True
    assert result.upload_receipt is not None
    assert result.upload_receipt["expected_chunk_count"] == 2

    uploaded_episode_dir = tmp_path / "spool" / "uploaded" / result.episode_id
    sink_episode_dir = tmp_path / "ingestion" / result.episode_id
    assert uploaded_episode_dir.exists()
    assert (sink_episode_dir / "commit.json").exists()

    with (uploaded_episode_dir / "upload_receipt.json").open("r", encoding="utf-8") as handle:
        saved_receipt = json.load(handle)
    assert saved_receipt["episode_id"] == result.episode_id


def test_run_edge_local_dry_run_with_http_upload(tmp_path):
    server, thread = _start_ingestion_server(tmp_path)
    try:
        cfg = EdgeRunLocalConfig(
            robot=MockRobotConfig(
                n_motors=3,
                random_values=False,
                static_values=[1.0, 2.0, 3.0],
            ),
            runtime=StaticActionRuntimeConfig(
                action_dim=3,
                actions_per_chunk=2,
                action_value=0.75,
            ),
            spool_root=str(tmp_path / "spool"),
            model_store_root=str(tmp_path / "models"),
            ingestion_base_url=f"http://127.0.0.1:{server.server_port}",
            max_steps=3,
            fps=20,
            dry_run=True,
            upload_after_run=True,
            upload_steps_per_chunk=2,
        )

        result = run_edge_local(cfg)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.step_count == 3
    assert result.uploaded is True
    assert result.upload_receipt is not None
    assert result.upload_receipt["expected_chunk_count"] == 2

    uploaded_episode_dir = tmp_path / "spool" / "uploaded" / result.episode_id
    sink_episode_dir = tmp_path / "ingestion" / result.episode_id
    assert uploaded_episode_dir.exists()
    assert (sink_episode_dir / "commit.json").exists()


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


def test_run_edge_local_resolves_artifact_from_registry(tmp_path):
    model_root = tmp_path / "models"
    registry_root = tmp_path / "registry"
    _write_artifact(model_root, "artifact-prod-001")

    registry = ReleaseRegistry(registry_root)
    registry.set_channel_target(channel="prod", target_artifact_id="artifact-prod-001")
    registry.assign_device(device_id="edge-01", channel="prod", robot_id="robot-01", task_id="pick")

    cfg = EdgeRunLocalConfig(
        robot=MockRobotConfig(
            n_motors=3,
            random_values=False,
            static_values=[1.0, 2.0, 3.0],
        ),
        runtime=StaticActionRuntimeConfig(
            action_dim=3,
            actions_per_chunk=2,
            action_value=0.75,
        ),
        spool_root=str(tmp_path / "spool"),
        model_store_root=str(model_root),
        registry_root=str(registry_root),
        device_id="edge-01",
        max_steps=1,
        fps=20,
        dry_run=True,
        use_action_processor=False,
    )

    result = run_edge_local(cfg)

    sealed_episode_dir = tmp_path / "spool" / "sealed" / result.episode_id
    with (sealed_episode_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["policy_artifact_id"] == "artifact-prod-001"


def test_run_edge_local_syncs_registry_target_per_episode_boundary(tmp_path):
    model_root = tmp_path / "models"
    registry_root = tmp_path / "registry"
    _write_artifact(model_root, "artifact-prod-001")
    _write_artifact(model_root, "artifact-prod-002")

    registry = ReleaseRegistry(registry_root)
    registry.set_channel_target(channel="prod", target_artifact_id="artifact-prod-001")
    registry.assign_device(device_id="edge-01", channel="prod", robot_id="robot-01", task_id="pick")

    cfg = EdgeRunLocalConfig(
        robot=MockRobotConfig(
            n_motors=3,
            random_values=False,
            static_values=[1.0, 2.0, 3.0],
        ),
        runtime=StaticActionRuntimeConfig(
            action_dim=3,
            actions_per_chunk=2,
            action_value=0.75,
        ),
        spool_root=str(tmp_path / "spool"),
        model_store_root=str(model_root),
        registry_root=str(registry_root),
        device_id="edge-01",
        num_episodes=2,
        max_steps=1,
        fps=20,
        dry_run=True,
        use_action_processor=False,
    )

    def before_episode(episode_idx: int) -> None:
        if episode_idx == 1:
            registry.set_channel_target(channel="prod", target_artifact_id="artifact-prod-002")

    result = run_edge_local(cfg, before_episode=before_episode)

    assert result.episodes_run == 2
    assert result.policy_artifact_ids == ["artifact-prod-001", "artifact-prod-002"]
    assert result.sync_actions == ["staged", "staged"]
