#!/usr/bin/env python

from __future__ import annotations

import json

from lerobot.cloud.ingestion import FilesystemEpisodeIngestionStore
from lerobot.cloud.materializer import FilesystemEpisodeMaterializer
from lerobot.edge.mock_runtime import StaticActionRuntime, StaticActionRuntimeConfig
from lerobot.edge.runner import EdgeRobotRunner
from lerobot.edge.spool import EdgeEpisodeSpool
from lerobot.edge.uploader import EdgeEpisodeUploader, EdgeUploaderConfig
from lerobot.robots.utils import make_robot_from_config
from tests.mocks.mock_robot import MockRobotConfig


def test_materializer_builds_dataset_manifest_from_committed_ingestion(tmp_path):
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
        runner = EdgeRobotRunner(
            robot=robot,
            runtime=runtime,
            spool=spool,
            policy_artifact_id="artifact-materializer",
            processor_bundle_id="processor-materializer",
            task_id="materialize-task",
        )
        result = runner.run_episode(max_steps=3, fps=20)
    finally:
        robot.disconnect()

    uploader = EdgeEpisodeUploader(
        spool=spool,
        sink=ingestion,
        config=EdgeUploaderConfig(steps_per_chunk=2),
    )
    uploader.upload_episode(result.episode_id)

    materializer = FilesystemEpisodeMaterializer(
        ingestion_root=tmp_path / "ingestion",
        output_root=tmp_path / "materialized",
    )
    manifest = materializer.materialize_all()

    assert manifest.episode_count == 1
    assert manifest.total_step_count == 3
    assert manifest.episodes[0].episode_id == result.episode_id
    assert manifest.episodes[0].policy_artifact_id == "artifact-materializer"

    materialized_episode_dir = tmp_path / "materialized" / "episodes" / result.episode_id
    assert (materialized_episode_dir / "steps.jsonl").exists()
    assert (tmp_path / "materialized" / "manifest.json").exists()

    with (materialized_episode_dir / "steps.jsonl").open("r", encoding="utf-8") as handle:
        steps = [json.loads(line) for line in handle]
    assert len(steps) == 3
    assert steps[0]["step_idx"] == 0
