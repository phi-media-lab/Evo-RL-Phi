#!/usr/bin/env python

from __future__ import annotations

import json

from lerobot.cloud.ingestion import FilesystemEpisodeIngestionStore
from lerobot.cloud.materializer import FilesystemEpisodeMaterializer
from lerobot.control_plane.registry import ReleaseRegistry
from lerobot.edge.mock_runtime import StaticActionRuntime, StaticActionRuntimeConfig
from lerobot.edge.runner import EdgeRobotRunner
from lerobot.edge.spool import EdgeEpisodeSpool
from lerobot.edge.uploader import EdgeEpisodeUploader, EdgeUploaderConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.scripts.control_plane_release_cycle import run_release_cycle
from tests.mocks.mock_robot import MockRobotConfig


def test_release_cycle_writes_rollout_report_snapshot(tmp_path):
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
        episode = EdgeRobotRunner(
            robot=robot,
            runtime=runtime,
            spool=spool,
            policy_artifact_id="release-cycle-policy",
            processor_bundle_id="release-cycle-processor",
            task_id="release-cycle-task",
        ).run_episode(max_steps=3, fps=20)
    finally:
        robot.disconnect()

    EdgeEpisodeUploader(
        spool=spool,
        sink=ingestion,
        config=EdgeUploaderConfig(steps_per_chunk=2),
    ).upload_episode(episode.episode_id)

    dataset_root = tmp_path / "dataset"
    FilesystemEpisodeMaterializer(
        ingestion_root=tmp_path / "ingestion",
        output_root=tmp_path / "materialized",
    ).materialize_to_lerobot_dataset(
        repo_id="local/release-cycle",
        dataset_root=dataset_root,
        fps=20,
    )

    registry = ReleaseRegistry(tmp_path / "registry")
    registry.assign_device(device_id="edge-01", channel="staging")

    model_state_root = tmp_path / "edge-01-models"
    model_state_root.mkdir()
    with (model_state_root / "state.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "active_artifact_id": None,
                "pending_artifact_id": None,
                "previous_active_artifact_id": None,
            },
            handle,
        )

    result = run_release_cycle(
        dataset_root=str(dataset_root),
        repo_id="local/release-cycle",
        train_output_dir=str(tmp_path / "train_out"),
        artifact_output_root=str(tmp_path / "artifacts"),
        registry_root=str(tmp_path / "registry"),
        artifact_id="artifact-release-cycle-001",
        base_checkpoint="release-cycle-001",
        channel="staging",
        incident_root=str(tmp_path / "incidents"),
        report_root=str(tmp_path / "reports"),
        compatible_robot_types=["mock_robot"],
        compatible_camera_layouts=["single_arm_mock"],
        rollout_reason="auto publish release cycle artifact",
        device_state_roots={"edge-01": str(model_state_root)},
    )

    with open(result["report_path"], "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["channels"][0]["channel"] == "staging"
    assert payload["channels"][0]["target_artifact_id"] == "artifact-release-cycle-001"
    assert payload["devices"][0]["device_id"] == "edge-01"
    assert payload["devices"][0]["target_artifact_id"] == "artifact-release-cycle-001"
