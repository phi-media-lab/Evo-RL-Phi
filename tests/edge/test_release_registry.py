#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

from lerobot.control_plane import ArtifactManifest, ReleaseRegistry
from lerobot.edge.model_manager import EdgeModelManager


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
        compatible_robot_types=["mock"],
        compatible_camera_layouts=["front_single_rgb"],
        eval_summary={"offline_score": 1.0},
        policy_config={"type": "act"},
        metadata={"policy_path": "policy"},
    )
    manifest.write_json(artifact_dir / "manifest.json")


def test_release_registry_resolves_model_manager_artifact_by_device(tmp_path):
    model_root = tmp_path / "models"
    registry_root = tmp_path / "registry"
    _write_artifact(model_root, "artifact-prod-001")

    registry = ReleaseRegistry(registry_root)
    registry.set_channel_target(
        channel="prod",
        target_artifact_id="artifact-prod-001",
        rollout_reason="promote stable candidate",
    )
    registry.assign_device(
        device_id="robot-edge-01",
        channel="prod",
        robot_id="robot-01",
        task_id="pick",
    )

    manager = EdgeModelManager(model_root)
    artifact = manager.resolve_channel_artifact(registry, device_id="robot-edge-01")

    assert artifact.manifest.artifact_id == "artifact-prod-001"
    assert artifact.policy_path == model_root / "artifact-prod-001" / "policy"


def test_release_registry_fallback_channel_without_assignment(tmp_path):
    model_root = tmp_path / "models"
    registry_root = tmp_path / "registry"
    _write_artifact(model_root, "artifact-staging-002")

    registry = ReleaseRegistry(registry_root)
    registry.set_channel_target(
        channel="staging",
        target_artifact_id="artifact-staging-002",
    )

    manager = EdgeModelManager(model_root)
    artifact = manager.resolve_channel_artifact(
        registry,
        device_id="unassigned-edge",
        fallback_channel="staging",
    )

    assert artifact.manifest.artifact_id == "artifact-staging-002"
