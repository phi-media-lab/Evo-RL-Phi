#!/usr/bin/env python

from __future__ import annotations

import json

from lerobot.cloud.ingestion import FilesystemEpisodeIngestionStore
from lerobot.cloud.materializer import FilesystemEpisodeMaterializer
from lerobot.control_plane.registry import ReleaseRegistry
from lerobot.control_plane.runner import AutoReleaseController, AutoReleaseDaemon
from lerobot.edge.mock_runtime import StaticActionRuntime, StaticActionRuntimeConfig
from lerobot.edge.runner import EdgeRobotRunner
from lerobot.edge.spool import EdgeEpisodeSpool
from lerobot.edge.uploader import EdgeEpisodeUploader, EdgeUploaderConfig
from lerobot.robots.utils import make_robot_from_config
from tests.mocks.mock_robot import MockRobotConfig


def _materialize_dataset(tmp_path):
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
            policy_artifact_id="auto-release-policy",
            processor_bundle_id="auto-release-processor",
            task_id="auto-release-task",
        ).run_episode(max_steps=3, fps=20)
    finally:
        robot.disconnect()

    EdgeEpisodeUploader(
        spool=spool,
        sink=ingestion,
        config=EdgeUploaderConfig(steps_per_chunk=2),
    ).upload_episode(episode.episode_id)

    dataset_root = tmp_path / "dataset"
    materialized_root = tmp_path / "materialized"
    FilesystemEpisodeMaterializer(
        ingestion_root=tmp_path / "ingestion",
        output_root=materialized_root,
    ).materialize_to_lerobot_dataset(
        repo_id="local/auto-release",
        dataset_root=dataset_root,
        fps=20,
    )
    return dataset_root, materialized_root


def test_auto_release_controller_releases_on_new_materialized_dataset(tmp_path):
    _, materialized_root = _materialize_dataset(tmp_path)

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

    result = AutoReleaseController(
        registry_root=tmp_path / "registry",
        artifact_output_root=tmp_path / "artifacts",
        state_root=tmp_path / "controller_state",
    ).run_once(
        materialized_root=materialized_root,
        train_output_root=tmp_path / "train_runs",
        incident_root=tmp_path / "incidents",
        report_root=tmp_path / "reports",
        channel="staging",
        artifact_prefix="artifact-auto-release",
        compatible_robot_types=["mock_robot"],
        compatible_camera_layouts=["single_arm_mock"],
        rollout_reason="auto promote materialized dataset",
        device_state_roots={"edge-01": str(model_state_root)},
    )

    channel_target = registry.get_channel_target("staging")

    assert result.action == "released"
    assert result.artifact_id is not None
    assert channel_target is not None
    assert channel_target.target_artifact_id == result.artifact_id


def test_auto_release_controller_noops_when_materialized_dataset_is_unchanged(tmp_path):
    _, materialized_root = _materialize_dataset(tmp_path)

    controller = AutoReleaseController(
        registry_root=tmp_path / "registry",
        artifact_output_root=tmp_path / "artifacts",
        state_root=tmp_path / "controller_state",
    )
    first = controller.run_once(
        materialized_root=materialized_root,
        train_output_root=tmp_path / "train_runs",
        incident_root=tmp_path / "incidents",
        report_root=tmp_path / "reports",
        channel="staging",
        artifact_prefix="artifact-auto-release",
        compatible_robot_types=["mock_robot"],
        compatible_camera_layouts=["single_arm_mock"],
        rollout_reason="auto promote materialized dataset",
    )
    second = controller.run_once(
        materialized_root=materialized_root,
        train_output_root=tmp_path / "train_runs",
        incident_root=tmp_path / "incidents",
        report_root=tmp_path / "reports",
        channel="staging",
        artifact_prefix="artifact-auto-release",
        compatible_robot_types=["mock_robot"],
        compatible_camera_layouts=["single_arm_mock"],
        rollout_reason="auto promote materialized dataset",
    )

    assert first.action == "released"
    assert second.action == "noop"
    assert second.artifact_id == first.artifact_id
    assert second.dataset_fingerprint == first.dataset_fingerprint


def test_auto_release_daemon_waits_when_no_manifest_exists(tmp_path):
    loop = AutoReleaseDaemon(
        registry_root=tmp_path / "registry",
        artifact_output_root=tmp_path / "artifacts",
        state_root=tmp_path / "controller_state",
        runtime_root=tmp_path / "controller_runtime",
    ).run(
        materialized_root=tmp_path / "materialized",
        train_output_root=tmp_path / "train_runs",
        incident_root=tmp_path / "incidents",
        report_root=tmp_path / "reports",
        channel="staging",
        artifact_prefix="artifact-auto-release",
        compatible_robot_types=["mock_robot"],
        compatible_camera_layouts=["single_arm_mock"],
        max_iterations=1,
        poll_interval_s=0.0,
        sleep_fn=lambda _: None,
    )

    assert loop.iterations == 1
    assert loop.results[0].action == "waiting"


def test_auto_release_daemon_records_released_then_noop(tmp_path):
    _, materialized_root = _materialize_dataset(tmp_path)

    loop = AutoReleaseDaemon(
        registry_root=tmp_path / "registry",
        artifact_output_root=tmp_path / "artifacts",
        state_root=tmp_path / "controller_state",
        runtime_root=tmp_path / "controller_runtime",
    ).run(
        materialized_root=materialized_root,
        train_output_root=tmp_path / "train_runs",
        incident_root=tmp_path / "incidents",
        report_root=tmp_path / "reports",
        channel="staging",
        artifact_prefix="artifact-auto-release",
        compatible_robot_types=["mock_robot"],
        compatible_camera_layouts=["single_arm_mock"],
        rollout_reason="auto promote materialized dataset",
        max_iterations=2,
        poll_interval_s=0.0,
        sleep_fn=lambda _: None,
    )

    history_path = tmp_path / "controller_runtime" / "history.jsonl"
    latest_path = tmp_path / "controller_runtime" / "latest.json"
    history_lines = history_path.read_text(encoding="utf-8").strip().splitlines()
    latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))

    assert loop.iterations == 2
    assert [result.action for result in loop.results] == ["released", "noop"]
    assert len(history_lines) == 2
    assert latest_payload["action"] == "noop"
