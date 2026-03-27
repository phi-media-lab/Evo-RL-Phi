#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

from lerobot.control_plane import ArtifactManifest, ReleaseRegistry
from lerobot.edge.model_manager import EdgeCompatibilityContext, EdgeModelManager


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


def test_model_manager_sync_stages_target_from_registry(tmp_path):
    model_root = tmp_path / "models"
    registry_root = tmp_path / "registry"
    _write_artifact(model_root, "artifact-staging-001")

    registry = ReleaseRegistry(registry_root)
    registry.set_channel_target(channel="staging", target_artifact_id="artifact-staging-001")
    registry.assign_device(device_id="edge-01", channel="staging")

    manager = EdgeModelManager(model_root)
    result = manager.sync_to_registry_target(
        registry,
        device_id="edge-01",
        context=EdgeCompatibilityContext(
            robot_type="mock_robot",
            camera_layout="front_single_rgb",
        ),
    )

    assert result.action == "staged"
    assert result.channel == "staging"
    assert result.target_artifact_id == "artifact-staging-001"
    assert result.changed is True
    assert manager.get_pending_artifact_id() == "artifact-staging-001"
    assert manager.get_active_artifact_id() is None


def test_model_manager_sync_noop_when_target_is_already_active(tmp_path):
    model_root = tmp_path / "models"
    registry_root = tmp_path / "registry"
    _write_artifact(model_root, "artifact-prod-001")

    registry = ReleaseRegistry(registry_root)
    registry.set_channel_target(channel="prod", target_artifact_id="artifact-prod-001")
    registry.assign_device(device_id="edge-01", channel="prod")

    manager = EdgeModelManager(model_root)
    manager.state.active_artifact_id = "artifact-prod-001"
    manager._write_state()

    result = manager.sync_to_registry_target(
        registry,
        device_id="edge-01",
        context=EdgeCompatibilityContext(
            robot_type="mock_robot",
            camera_layout="front_single_rgb",
        ),
    )

    assert result.action == "noop"
    assert result.changed is False
    assert manager.get_active_artifact_id() == "artifact-prod-001"


def test_model_manager_rollback_to_previous_active(tmp_path):
    model_root = tmp_path / "models"
    _write_artifact(model_root, "artifact-prod-001")
    _write_artifact(model_root, "artifact-prod-002")

    manager = EdgeModelManager(model_root)
    manager.state.active_artifact_id = "artifact-prod-001"
    manager.state.previous_active_artifact_id = None
    manager.state.pending_artifact_id = "artifact-prod-002"
    manager._write_state()

    from lerobot.edge.runtime import LocalPolicyRuntime

    class DummyRuntime(LocalPolicyRuntime):
        def __init__(self):
            pass

    original_activate_artifact = manager.activate_artifact

    def fake_activate_artifact(artifact, *, runtime_overrides=None, warmup_observation=None):
        if manager.state.active_artifact_id != artifact.manifest.artifact_id:
            manager.state.previous_active_artifact_id = manager.state.active_artifact_id
        manager.state.active_artifact_id = artifact.manifest.artifact_id
        manager.state.pending_artifact_id = None
        manager._write_state()
        return DummyRuntime()

    manager.activate_artifact = fake_activate_artifact
    try:
        manager.state.previous_active_artifact_id = "artifact-prod-001"
        runtime = manager.rollback_to_previous_active(runtime_overrides={"device": "cpu"})
    finally:
        manager.activate_artifact = original_activate_artifact

    assert isinstance(runtime, DummyRuntime)
    assert manager.get_active_artifact_id() == "artifact-prod-001"
    assert manager.get_pending_artifact_id() is None
