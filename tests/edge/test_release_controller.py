#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

from lerobot.cloud.artifact_builder import ArtifactBuildRequest, FilesystemArtifactBuilder
from lerobot.control_plane.controller import ReleaseController
from lerobot.control_plane.registry import ReleaseRegistry


def _write_policy_dir(root: Path) -> Path:
    policy_dir = root / "policy"
    policy_dir.mkdir(parents=True, exist_ok=False)
    (policy_dir / "config.json").write_text('{"type":"act"}\n', encoding="utf-8")
    (policy_dir / "model.safetensors").write_bytes(b"stub")
    return policy_dir


def test_release_controller_publishes_artifact_to_channel(tmp_path):
    policy_dir = _write_policy_dir(tmp_path)
    artifact_root = FilesystemArtifactBuilder().build(
        ArtifactBuildRequest(
            artifact_id="artifact-release-001",
            base_checkpoint="ckpt-1",
            policy_dir=policy_dir,
            output_root=tmp_path / "artifacts",
            compatible_robot_types=["mock_robot"],
            compatible_camera_layouts=["single_arm_mock"],
        )
    )

    result = ReleaseController(
        registry_root=tmp_path / "registry",
        artifact_root=tmp_path / "artifacts",
    ).publish_artifact_to_channel(
        channel="staging",
        artifact_id="artifact-release-001",
        rollout_reason="promote train-and-build artifact",
    )

    registry = ReleaseRegistry(tmp_path / "registry")
    channel_target = registry.get_channel_target("staging")

    assert artifact_root.exists()
    assert result.channel == "staging"
    assert result.artifact_id == "artifact-release-001"
    assert channel_target is not None
    assert channel_target.target_artifact_id == "artifact-release-001"
    assert channel_target.rollout_reason == "promote train-and-build artifact"
