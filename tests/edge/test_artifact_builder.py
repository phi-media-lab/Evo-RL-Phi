#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

from lerobot.cloud.artifact_builder import (
    ArtifactBuildRequest,
    FilesystemArtifactBuilder,
    FilesystemOpenPIArtifactBuilder,
    OpenPIArtifactBuildRequest,
)
from lerobot.cloud.ingestion import FilesystemEpisodeIngestionStore
from lerobot.cloud.materializer import FilesystemEpisodeMaterializer
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.edge.mock_runtime import StaticActionRuntime, StaticActionRuntimeConfig
from lerobot.edge.model_manager import EdgeCompatibilityContext, EdgeModelManager
from lerobot.edge.openpi_runtime import OpenPIInProcessRuntimeConfig
from lerobot.edge.runner import EdgeRobotRunner
from lerobot.edge.spool import EdgeEpisodeSpool
from lerobot.edge.uploader import EdgeEpisodeUploader, EdgeUploaderConfig
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.robots.utils import make_robot_from_config
from tests.mocks.mock_robot import MockRobotConfig


def _create_saved_policy_dir(tmp_path: Path) -> tuple[Path, Path]:
    robot = make_robot_from_config(
        MockRobotConfig(
            n_motors=3,
            random_values=False,
            static_values=[1.0, 2.0, 3.0],
        )
    )
    runtime = StaticActionRuntime(
        StaticActionRuntimeConfig(
            action_dim=3,
            actions_per_chunk=2,
            action_value=0.5,
        )
    )
    spool = EdgeEpisodeSpool(tmp_path / "spool")
    ingestion = FilesystemEpisodeIngestionStore(tmp_path / "ingestion")

    robot.connect()
    try:
        result = EdgeRobotRunner(
            robot=robot,
            runtime=runtime,
            spool=spool,
            policy_artifact_id="artifact-builder-policy",
            processor_bundle_id="artifact-builder-processor",
            task_id="artifact-builder-task",
        ).run_episode(max_steps=3, fps=20)
    finally:
        robot.disconnect()

    EdgeEpisodeUploader(
        spool=spool,
        sink=ingestion,
        config=EdgeUploaderConfig(steps_per_chunk=2),
    ).upload_episode(result.episode_id)

    dataset_root = tmp_path / "lerobot_dataset"
    FilesystemEpisodeMaterializer(
        ingestion_root=tmp_path / "ingestion",
        output_root=tmp_path / "materialized",
    ).materialize_to_lerobot_dataset(
        repo_id="local/artifact-builder",
        dataset_root=dataset_root,
        fps=20,
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="local/artifact-builder", root=dataset_root),
        policy=make_policy_config("act", push_to_hub=False, device="cpu"),
        output_dir=tmp_path / "train_out",
        steps=1,
        batch_size=1,
        num_workers=0,
        eval_freq=999999,
        save_freq=999999,
        log_freq=1,
    )
    cfg.validate()
    dataset = make_dataset(cfg)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)

    policy_dir = tmp_path / "pretrained_policy"
    policy.save_pretrained(policy_dir)
    return policy_dir, dataset_root / "meta" / "stats.json"


def test_artifact_builder_produces_model_manager_compatible_artifact(tmp_path):
    policy_dir, stats_path = _create_saved_policy_dir(tmp_path)

    builder = FilesystemArtifactBuilder()
    artifact_root = builder.build(
        ArtifactBuildRequest(
            artifact_id="artifact-act-001",
            base_checkpoint="train-smoke-001",
            policy_dir=policy_dir,
            output_root=tmp_path / "artifacts",
            compatible_robot_types=["mock_robot"],
            compatible_camera_layouts=["single_arm_mock"],
            stats_path=stats_path,
            eval_summary={"loss": 1.23},
            metadata={"task": "artifact-builder-task", "robot_type": "mock_robot"},
        )
    )

    assert (artifact_root / "manifest.json").exists()
    assert (artifact_root / "policy" / "config.json").exists()
    assert (artifact_root / "stats.json").exists()

    model_manager = EdgeModelManager(tmp_path / "artifacts")
    managed_artifact = model_manager.load_artifact_by_id("artifact-act-001")
    errors = model_manager.validate_artifact(
        managed_artifact,
        EdgeCompatibilityContext(
            robot_type="mock_robot",
            camera_layout="single_arm_mock",
        ),
    )

    assert errors == []
    assert managed_artifact.manifest.artifact_id == "artifact-act-001"
    assert managed_artifact.manifest.policy_config["type"] == "act"
    assert managed_artifact.policy_path == artifact_root / "policy"


def test_openpi_artifact_builder_produces_model_manager_compatible_artifact(tmp_path):
    runtime_assets_dir = tmp_path / "runtime_assets_src"
    runtime_assets_dir.mkdir()
    (runtime_assets_dir / "sample_observation.json").write_text('{"prompt":"Transfer cube"}\n', encoding="utf-8")

    artifact_root = FilesystemOpenPIArtifactBuilder().build(
        OpenPIArtifactBuildRequest(
            artifact_id="artifact-openpi-aloha-001",
            base_checkpoint="pi0_aloha_sim",
            output_root=tmp_path / "artifacts",
            compatible_robot_types=["mock_robot"],
            compatible_camera_layouts=["single_arm_mock"],
            policy_ref="openpi://checkpoint?config=pi0_aloha_sim&dir=/tmp/pi0_aloha_sim",
            runtime_backend="in_process",
            action_horizon=50,
            action_dim=14,
            model_output_key="rollout",
            output_transform="aloha_actions",
            runtime_metadata={
                "bridge_mode": "mlx_full_prefix_hostbridge",
                "config_name": "pi0_aloha_sim",
                "checkpoint_dir": "/tmp/pi0_aloha_sim_pytorch",
            },
            observation_contract={
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
            runtime_assets_dir=runtime_assets_dir,
            metadata={"task": "aloha_transfer_cube", "robot_type": "mock_robot"},
        )
    )

    assert (artifact_root / "manifest.json").exists()
    assert (artifact_root / "policy" / "config.json").exists()
    assert (artifact_root / "runtime_assets" / "sample_observation.json").exists()

    model_manager = EdgeModelManager(tmp_path / "artifacts")
    managed_artifact = model_manager.load_artifact_by_id("artifact-openpi-aloha-001")
    errors = model_manager.validate_artifact(
        managed_artifact,
        EdgeCompatibilityContext(
            robot_type="mock_robot",
            camera_layout="single_arm_mock",
        ),
    )
    runtime_config = model_manager.build_runtime_config(managed_artifact)

    assert errors == []
    assert managed_artifact.manifest.policy_config["type"] == "openpi"
    assert managed_artifact.manifest.metadata["runtime"]["runtime_assets_path"] == "runtime_assets"
    assert managed_artifact.manifest.metadata["runtime"]["model_output_key"] == "rollout"
    assert managed_artifact.manifest.metadata["runtime"]["output_transform"] == "aloha_actions"
    assert managed_artifact.manifest.metadata["runtime"]["bridge_mode"] == "mlx_full_prefix_hostbridge"
    assert isinstance(runtime_config, OpenPIInProcessRuntimeConfig)
    assert runtime_config.observation.mode == "aloha_raw"
    assert runtime_config.action_dim == 14
