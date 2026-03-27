#!/usr/bin/env python

from __future__ import annotations

from lerobot.cloud.ingestion import FilesystemEpisodeIngestionStore
from lerobot.cloud.materializer import FilesystemEpisodeMaterializer
from lerobot.control_plane.registry import ReleaseRegistry
from lerobot.edge.mock_runtime import StaticActionRuntime, StaticActionRuntimeConfig
from lerobot.edge.model_manager import EdgeCompatibilityContext, EdgeModelManager
from lerobot.edge.runner import EdgeRobotRunner
from lerobot.edge.spool import EdgeEpisodeSpool
from lerobot.edge.uploader import EdgeEpisodeUploader, EdgeUploaderConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.scripts.cloud_train_build_and_release import run_train_build_and_release
from tests.mocks.mock_robot import MockRobotConfig


def test_train_build_and_release_updates_channel_registry(tmp_path):
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
            policy_artifact_id="train-build-release-policy",
            processor_bundle_id="train-build-release-processor",
            task_id="train-build-release-task",
        ).run_episode(max_steps=3, fps=20)
    finally:
        robot.disconnect()

    EdgeEpisodeUploader(
        spool=spool,
        sink=ingestion,
        config=EdgeUploaderConfig(steps_per_chunk=2),
    ).upload_episode(result.episode_id)

    dataset_root = tmp_path / "dataset"
    FilesystemEpisodeMaterializer(
        ingestion_root=tmp_path / "ingestion",
        output_root=tmp_path / "materialized",
    ).materialize_to_lerobot_dataset(
        repo_id="local/train-build-release",
        dataset_root=dataset_root,
        fps=20,
    )

    release_result = run_train_build_and_release(
        dataset_root=str(dataset_root),
        repo_id="local/train-build-release",
        train_output_dir=str(tmp_path / "train_out"),
        artifact_output_root=str(tmp_path / "artifacts"),
        registry_root=str(tmp_path / "registry"),
        artifact_id="artifact-train-build-release-001",
        base_checkpoint="train-build-release-001",
        channel="staging",
        rollout_reason="auto publish train-and-build artifact",
        compatible_robot_types=["mock_robot"],
        compatible_camera_layouts=["single_arm_mock"],
    )

    registry = ReleaseRegistry(tmp_path / "registry")
    channel_target = registry.get_channel_target("staging")
    model_manager = EdgeModelManager(tmp_path / "artifacts")
    artifact = model_manager.load_artifact_by_id("artifact-train-build-release-001")
    errors = model_manager.validate_artifact(
        artifact,
        EdgeCompatibilityContext(
            robot_type="mock_robot",
            camera_layout="single_arm_mock",
        ),
    )

    assert errors == []
    assert release_result["channel"] == "staging"
    assert release_result["artifact_id"] == "artifact-train-build-release-001"
    assert channel_target is not None
    assert channel_target.target_artifact_id == "artifact-train-build-release-001"
    assert channel_target.rollout_reason == "auto publish train-and-build artifact"
