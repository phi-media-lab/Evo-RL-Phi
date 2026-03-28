#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

from lerobot.control_plane import ArtifactManifest, ReleaseRegistry
from lerobot.edge.model_manager import EdgeCompatibilityContext, EdgeModelManager
from lerobot.edge.openpi_runtime import (
    OpenPICoreMLRuntimeConfig,
    OpenPICoreMLRuntimeFactory,
    OpenPIHybridBridgeRuntimeConfig,
    OpenPIHybridBridgeRuntimeFactory,
    OpenPIInProcessRuntimeConfig,
    OpenPIWebsocketRuntimeConfig,
)
from lerobot.edge.runtime_protocol import EdgePolicyRuntime


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


def _write_openpi_artifact(root: Path, artifact_id: str) -> None:
    artifact_dir = root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "policy").mkdir()
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        base_checkpoint="ckpt-openpi-1",
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
        policy_config={"type": "openpi"},
        metadata={
            "policy_path": "policy",
            "runtime": {
                "backend": "websocket",
                "server_uri": "ws://127.0.0.1:8000",
                "action_horizon": 8,
                "action_dim": 3,
                "openpi_client_root": "/tmp/openpi-client",
            },
            "observation_contract": {
                "state_keys": ["motor_1.pos", "motor_2.pos", "motor_3.pos"],
                "prompt": "pick",
            },
        },
    )
    manifest.write_json(artifact_dir / "manifest.json")


def _write_openpi_in_process_artifact(root: Path, artifact_id: str) -> None:
    artifact_dir = root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "policy").mkdir()
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        base_checkpoint="ckpt-openpi-2",
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
        policy_config={"type": "openpi"},
        metadata={
            "policy_path": "policy",
            "runtime": {
                "backend": "in_process",
                "action_horizon": 8,
                "action_dim": 3,
                "policy_ref": "openpi://pi0_aloha_sim",
            },
            "observation_contract": {
                "state_keys": ["motor_1.pos", "motor_2.pos", "motor_3.pos"],
                "prompt": "pick",
            },
        },
    )
    manifest.write_json(artifact_dir / "manifest.json")


def _write_openpi_aloha_artifact(root: Path, artifact_id: str) -> None:
    artifact_dir = root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "policy").mkdir()
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        base_checkpoint="ckpt-openpi-aloha-1",
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
        policy_config={"type": "openpi"},
        metadata={
            "policy_path": "policy",
            "runtime": {
                "backend": "in_process",
                "action_horizon": 50,
                "action_dim": 14,
                "policy_ref": "openpi://checkpoint?config=pi0_aloha_sim&dir=/tmp/pi0_aloha_sim",
            },
            "observation_contract": {
                "mode": "aloha_raw",
                "state_key": "state",
                "state_dim": 14,
                "image_keys": [
                    "image.base_0_rgb",
                    "image.left_wrist_0_rgb",
                    "image.right_wrist_0_rgb",
                ],
                "image_aliases": {
                    "image.base_0_rgb": "cam_high",
                    "image.left_wrist_0_rgb": "cam_left_wrist",
                    "image.right_wrist_0_rgb": "cam_right_wrist",
                },
                "prompt_key": "prompt",
                "transpose_images_to_chw": True,
            },
        },
    )
    manifest.write_json(artifact_dir / "manifest.json")


def _write_openpi_coreml_artifact(root: Path, artifact_id: str, *, with_runtime_assets: bool = True) -> None:
    artifact_dir = root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "policy").mkdir()
    if with_runtime_assets:
        runtime_assets_dir = artifact_dir / "runtime_assets"
        runtime_assets_dir.mkdir()
        (runtime_assets_dir / "model.mlpackage").mkdir()
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        base_checkpoint="ckpt-openpi-coreml-1",
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
        policy_config={"type": "openpi"},
        metadata={
            "policy_path": "policy",
            "runtime": {
                "backend": "coreml",
                "runtime_assets_path": "runtime_assets",
                "action_horizon": 8,
                "action_dim": 14,
                "model_output_key": "rollout",
                "output_transform": "aloha_actions",
                "compute_unit": "cpu_and_ne",
            },
            "observation_contract": {
                "mode": "aloha_raw",
                "state_key": "state",
                "state_dim": 14,
                "image_keys": ["image.base_0_rgb"],
                "image_aliases": {"image.base_0_rgb": "cam_high"},
                "transpose_images_to_chw": True,
            },
        },
    )
    manifest.write_json(artifact_dir / "manifest.json")


def _write_openpi_hybrid_artifact(root: Path, artifact_id: str) -> None:
    artifact_dir = root / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "policy").mkdir()
    runtime_assets_dir = artifact_dir / "runtime_assets"
    runtime_assets_dir.mkdir()
    (runtime_assets_dir / "model.mlpackage").mkdir()
    (runtime_assets_dir / "feed_contract.json").write_text('{"bridge":"v1"}\n', encoding="utf-8")
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        base_checkpoint="ckpt-openpi-hybrid-1",
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
        policy_config={"type": "openpi"},
        metadata={
            "policy_path": "policy",
            "runtime": {
                "backend": "hybrid_bridge_v1",
                "runtime_assets_path": "runtime_assets",
                "bridge_mode": "mlx_full_prefix_hostbridge",
                "prefix_mode": "mlx_prefix_v1",
                "action_horizon": 8,
                "action_dim": 14,
                "model_output_key": "rollout",
                "output_transform": "aloha_actions",
                "compute_unit": "cpu_and_ne",
            },
            "observation_contract": {
                "mode": "aloha_raw",
                "state_key": "state",
                "state_dim": 14,
                "image_keys": ["image.base_0_rgb"],
                "image_aliases": {"image.base_0_rgb": "cam_high"},
                "transpose_images_to_chw": True,
            },
        },
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


def test_model_manager_builds_openpi_runtime_config(tmp_path):
    model_root = tmp_path / "models"
    _write_openpi_artifact(model_root, "artifact-openpi-001")

    manager = EdgeModelManager(model_root)
    artifact = manager.load_artifact_by_id("artifact-openpi-001")
    runtime_config = manager.build_runtime_config(artifact)

    assert isinstance(runtime_config, OpenPIWebsocketRuntimeConfig)
    assert runtime_config.server_uri == "ws://127.0.0.1:8000"
    assert runtime_config.action_horizon == 8
    assert runtime_config.action_dim == 3
    assert runtime_config.observation.state_keys == ["motor_1.pos", "motor_2.pos", "motor_3.pos"]
    assert runtime_config.observation.prompt == "pick"


def test_model_manager_activates_openpi_artifact_with_custom_factory(tmp_path):
    class FakeRuntime:
        queue_size = 0

        def reset(self) -> None:
            return None

        def warmup(self, observation=None):
            return None

        def maybe_refill_action_queue(self, observation=None):
            return None

        def refill_action_queue(self, observation=None):
            return None

        def pop_next_action(self, observation=None):
            raise NotImplementedError

    class FakeRuntimeFactory:
        runtime_config_type = OpenPIWebsocketRuntimeConfig

        def __init__(self):
            self.last_runtime_config = None

        def build_runtime(self, runtime_config, *, warmup_observation=None) -> EdgePolicyRuntime:
            self.last_runtime_config = runtime_config
            return FakeRuntime()

        def supports_runtime(self, runtime: EdgePolicyRuntime) -> bool:
            return isinstance(runtime, FakeRuntime)

    model_root = tmp_path / "models"
    _write_openpi_artifact(model_root, "artifact-openpi-001")
    runtime_factory = FakeRuntimeFactory()
    manager = EdgeModelManager(model_root, runtime_factory=runtime_factory)
    artifact = manager.load_artifact_by_id("artifact-openpi-001")

    runtime = manager.activate_artifact(artifact)

    assert isinstance(runtime, FakeRuntime)
    assert isinstance(runtime_factory.last_runtime_config, OpenPIWebsocketRuntimeConfig)
    assert manager.get_active_artifact_id() == "artifact-openpi-001"


def test_model_manager_builds_openpi_in_process_runtime_config(tmp_path):
    model_root = tmp_path / "models"
    _write_openpi_in_process_artifact(model_root, "artifact-openpi-inproc-001")

    manager = EdgeModelManager(model_root)
    artifact = manager.load_artifact_by_id("artifact-openpi-inproc-001")
    runtime_config = manager.build_runtime_config(artifact)

    assert isinstance(runtime_config, OpenPIInProcessRuntimeConfig)
    assert runtime_config.policy_ref == "openpi://pi0_aloha_sim"
    assert runtime_config.action_horizon == 8
    assert runtime_config.action_dim == 3


def test_model_manager_builds_openpi_aloha_observation_contract(tmp_path):
    model_root = tmp_path / "models"
    _write_openpi_aloha_artifact(model_root, "artifact-openpi-aloha-001")

    manager = EdgeModelManager(model_root)
    artifact = manager.load_artifact_by_id("artifact-openpi-aloha-001")
    runtime_config = manager.build_runtime_config(artifact)

    assert isinstance(runtime_config, OpenPIInProcessRuntimeConfig)
    assert runtime_config.observation.mode == "aloha_raw"
    assert runtime_config.observation.state_key == "state"
    assert runtime_config.observation.state_dim == 14
    assert runtime_config.observation.prompt_key == "prompt"
    assert runtime_config.observation.transpose_images_to_chw is True
    assert runtime_config.observation.image_aliases["image.base_0_rgb"] == "cam_high"


def test_model_manager_default_factory_activates_openpi_in_process_artifact(tmp_path, monkeypatch):
    class FakeRuntime:
        queue_size = 0

        def reset(self) -> None:
            return None

        def warmup(self, observation=None):
            return None

        def maybe_refill_action_queue(self, observation=None):
            return None

        def refill_action_queue(self, observation=None):
            return None

        def pop_next_action(self, observation=None):
            raise NotImplementedError

    model_root = tmp_path / "models"
    _write_openpi_in_process_artifact(model_root, "artifact-openpi-inproc-001")

    monkeypatch.setattr("lerobot.edge.model_manager.resolve_openpi_policy_ref", lambda policy_ref: object())
    monkeypatch.setattr(
        "lerobot.edge.openpi_runtime.OpenPIInProcessRuntimeFactory.build_runtime",
        lambda self, runtime_config, *, warmup_observation=None: FakeRuntime(),
    )

    manager = EdgeModelManager(model_root)
    artifact = manager.load_artifact_by_id("artifact-openpi-inproc-001")
    runtime = manager.activate_artifact(artifact)

    assert isinstance(runtime, FakeRuntime)
    assert manager.get_active_artifact_id() == "artifact-openpi-inproc-001"


def test_model_manager_builds_openpi_coreml_runtime_config(tmp_path):
    model_root = tmp_path / "models"
    _write_openpi_coreml_artifact(model_root, "artifact-openpi-coreml-001")

    manager = EdgeModelManager(model_root)
    artifact = manager.load_artifact_by_id("artifact-openpi-coreml-001")
    runtime_config = manager.build_runtime_config(artifact)

    assert isinstance(runtime_config, OpenPICoreMLRuntimeConfig)
    assert runtime_config.model_path.endswith("runtime_assets/model.mlpackage")
    assert runtime_config.compute_unit == "cpu_and_ne"
    assert runtime_config.action_dim == 14
    assert runtime_config.model_output_key == "rollout"
    assert runtime_config.output_transform == "aloha_actions"


def test_model_manager_builds_openpi_hybrid_runtime_config(tmp_path):
    model_root = tmp_path / "models"
    _write_openpi_hybrid_artifact(model_root, "artifact-openpi-hybrid-001")

    manager = EdgeModelManager(model_root)
    artifact = manager.load_artifact_by_id("artifact-openpi-hybrid-001")
    runtime_config = manager.build_runtime_config(artifact)

    assert isinstance(runtime_config, OpenPIHybridBridgeRuntimeConfig)
    assert runtime_config.model_path.endswith("runtime_assets/model.mlpackage")
    assert runtime_config.feed_contract_path.endswith("runtime_assets/feed_contract.json")
    assert runtime_config.bridge_mode == "mlx_full_prefix_hostbridge"
    assert runtime_config.prefix_mode == "mlx_prefix_v1"
    assert runtime_config.model_output_key == "rollout"
    assert runtime_config.output_transform == "aloha_actions"


def test_model_manager_discovers_single_runtime_mlpackage(tmp_path):
    model_root = tmp_path / "models"
    artifact_dir = model_root / "artifact-openpi-hybrid-custom-001"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "policy").mkdir()
    runtime_assets_dir = artifact_dir / "runtime_assets"
    runtime_assets_dir.mkdir()
    (runtime_assets_dir / "rollout8_custom.mlpackage").mkdir()
    manifest = ArtifactManifest(
        artifact_id="artifact-openpi-hybrid-custom-001",
        base_checkpoint="ckpt-openpi-hybrid-custom-1",
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
        policy_config={"type": "openpi"},
        metadata={
            "policy_path": "policy",
            "runtime": {
                "backend": "hybrid_bridge_v1",
                "runtime_assets_path": "runtime_assets",
                "bridge_mode": "mlx_full_prefix_hostbridge",
                "prefix_mode": "observation",
                "action_horizon": 8,
                "action_dim": 14,
            },
            "observation_contract": {
                "mode": "aloha_raw",
                "state_key": "state",
                "state_dim": 14,
                "image_keys": ["image.base_0_rgb"],
                "image_aliases": {"image.base_0_rgb": "cam_high"},
                "transpose_images_to_chw": True,
            },
        },
    )
    manifest.write_json(artifact_dir / "manifest.json")

    manager = EdgeModelManager(model_root)
    artifact = manager.load_artifact_by_id("artifact-openpi-hybrid-custom-001")
    runtime_config = manager.build_runtime_config(artifact)

    assert isinstance(runtime_config, OpenPIHybridBridgeRuntimeConfig)
    assert runtime_config.model_path.endswith("runtime_assets/rollout8_custom.mlpackage")


def test_model_manager_validate_openpi_runtime_assets_presence(tmp_path):
    model_root = tmp_path / "models"
    _write_openpi_coreml_artifact(model_root, "artifact-openpi-coreml-missing-001", with_runtime_assets=False)

    manager = EdgeModelManager(model_root)
    artifact = manager.load_artifact_by_id("artifact-openpi-coreml-missing-001")
    errors = manager.validate_artifact(
        artifact,
        EdgeCompatibilityContext(
            robot_type="mock_robot",
            camera_layout="front_single_rgb",
        ),
    )

    assert any("runtime assets path does not exist" in error for error in errors)


def test_model_manager_activates_openpi_hybrid_artifact_with_custom_factory(tmp_path):
    class FakeRuntime:
        queue_size = 0

        def reset(self) -> None:
            return None

        def warmup(self, observation=None):
            return None

        def maybe_refill_action_queue(self, observation=None):
            return None

        def refill_action_queue(self, observation=None):
            return None

        def pop_next_action(self, observation=None):
            raise NotImplementedError

    model_root = tmp_path / "models"
    _write_openpi_hybrid_artifact(model_root, "artifact-openpi-hybrid-001")
    runtime_factory = OpenPIHybridBridgeRuntimeFactory(predictor_factory=lambda runtime_config: (lambda feed: {"actions": [[0.0] * 14]}))
    manager = EdgeModelManager(model_root, runtime_factory=runtime_factory)
    artifact = manager.load_artifact_by_id("artifact-openpi-hybrid-001")

    runtime = manager.activate_artifact(artifact)

    assert manager.get_active_artifact_id() == "artifact-openpi-hybrid-001"
    assert runtime.__class__.__name__ == "OpenPIHybridBridgeRuntime"


def test_model_manager_activates_openpi_coreml_artifact_with_custom_factory(tmp_path):
    model_root = tmp_path / "models"
    _write_openpi_coreml_artifact(model_root, "artifact-openpi-coreml-001")
    runtime_factory = OpenPICoreMLRuntimeFactory(predictor_factory=lambda runtime_config: (lambda feed: {"actions": [[0.0] * 14]}))
    manager = EdgeModelManager(model_root, runtime_factory=runtime_factory)
    artifact = manager.load_artifact_by_id("artifact-openpi-coreml-001")

    runtime = manager.activate_artifact(artifact)

    assert manager.get_active_artifact_id() == "artifact-openpi-coreml-001"
    assert runtime.__class__.__name__ == "OpenPICoreMLRuntime"
