#!/usr/bin/env python

from __future__ import annotations

import json

from lerobot.edge.mock_runtime import StaticActionRuntime, StaticActionRuntimeConfig
from lerobot.edge.runner import EdgeRobotRunner
from lerobot.edge.spool import EdgeEpisodeSpool
from lerobot.robots.utils import make_robot_from_config
from tests.mocks.mock_robot import MockRobotConfig


def test_edge_runner_dry_run_records_uploaded_episode(tmp_path):
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
            actions_per_chunk=4,
            action_value=0.25,
        )
    )
    spool = EdgeEpisodeSpool(tmp_path / "spool")

    robot.connect()
    try:
        runner = EdgeRobotRunner(
            robot=robot,
            runtime=runtime,
            spool=spool,
            policy_artifact_id="dry-run-artifact",
            processor_bundle_id="dry-run-processor",
            task_id="dry-run-task",
            channel="dev",
            action_processor=None,
        )
        result = runner.run_episode(max_steps=3, fps=20)
        assert result.step_count == 3
        sealed_episode_dir = tmp_path / "spool" / "sealed" / result.episode_id
        assert sealed_episode_dir.exists()

        with (sealed_episode_dir / "summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        assert summary["policy_artifact_id"] == "dry-run-artifact"
        assert summary["stop_reason"] == "completed"

        with (sealed_episode_dir / "steps.jsonl").open("r", encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle]
        assert len(lines) == 3
        assert lines[0]["action"] == {"motor_1.pos": 0.25, "motor_2.pos": 0.25, "motor_3.pos": 0.25}
    finally:
        robot.disconnect()
