#!/usr/bin/env python

from __future__ import annotations

import json

from lerobot.cloud.ingestion import FilesystemEpisodeIngestionStore
from lerobot.edge.mock_runtime import StaticActionRuntime, StaticActionRuntimeConfig
from lerobot.edge.runner import EdgeRobotRunner
from lerobot.edge.spool import EdgeEpisodeSpool
from lerobot.edge.uploader import EdgeEpisodeUploader, EdgeUploaderConfig
from lerobot.robots.utils import make_robot_from_config
from tests.mocks.mock_robot import MockRobotConfig


def test_edge_uploader_moves_sealed_episode_to_uploaded(tmp_path):
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
    sink = FilesystemEpisodeIngestionStore(tmp_path / "ingestion")

    robot.connect()
    try:
        runner = EdgeRobotRunner(
            robot=robot,
            runtime=runtime,
            spool=spool,
            policy_artifact_id="uploader-artifact",
            processor_bundle_id="uploader-processor",
            task_id="uploader-task",
        )
        result = runner.run_episode(max_steps=3, fps=20)
    finally:
        robot.disconnect()

    uploader = EdgeEpisodeUploader(
        spool=spool,
        sink=sink,
        config=EdgeUploaderConfig(steps_per_chunk=2),
    )
    receipt = uploader.upload_episode(result.episode_id)

    uploaded_episode_dir = tmp_path / "spool" / "uploaded" / result.episode_id
    assert uploaded_episode_dir.exists()
    assert receipt["status"] == "committed"
    assert receipt["expected_chunk_count"] == 2
    assert len(receipt["chunks"]) == 2

    sink_episode_dir = tmp_path / "ingestion" / result.episode_id
    assert (sink_episode_dir / "commit.json").exists()
    assert (sink_episode_dir / "receipt.json").exists()
    assert (sink_episode_dir / "chunks" / "000000.jsonl").exists()
    assert (sink_episode_dir / "chunks" / "000001.jsonl").exists()

    with (uploaded_episode_dir / "upload_receipt.json").open("r", encoding="utf-8") as handle:
        saved_receipt = json.load(handle)
    assert saved_receipt["episode_id"] == result.episode_id
