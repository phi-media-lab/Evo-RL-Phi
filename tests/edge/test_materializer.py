#!/usr/bin/env python

from __future__ import annotations

import json

from lerobot.cloud.ingestion import FilesystemEpisodeIngestionStore
from lerobot.cloud.materializer import FilesystemEpisodeMaterializer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.edge.mock_runtime import StaticActionRuntime, StaticActionRuntimeConfig
from lerobot.edge.runner import EdgeRobotRunner
from lerobot.edge.spool import EdgeEpisodeSpool
from lerobot.edge.uploader import EdgeEpisodeUploader, EdgeUploaderConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.scripts.cloud_train_smoke import run_training_smoke
from lerobot.utils.constants import ACTION, DONE, OBS_ENV_STATE, OBS_STATE, REWARD
from tests.mocks.mock_robot import MockRobotConfig


def _build_committed_episode(tmp_path):
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
    return result


def test_materializer_builds_dataset_manifest_from_committed_ingestion(tmp_path):
    result = _build_committed_episode(tmp_path)

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


def test_materializer_exports_local_lerobot_dataset(tmp_path):
    _build_committed_episode(tmp_path)

    materializer = FilesystemEpisodeMaterializer(
        ingestion_root=tmp_path / "ingestion",
        output_root=tmp_path / "materialized",
    )
    manifest = materializer.materialize_to_lerobot_dataset(
        repo_id="local/mock-edge",
        dataset_root=tmp_path / "lerobot_dataset",
        fps=20,
    )

    assert manifest.episode_count == 1
    assert manifest.total_step_count == 3
    assert manifest.lerobot_repo_id == "local/mock-edge"
    assert manifest.lerobot_root == str(tmp_path / "lerobot_dataset")

    dataset = LeRobotDataset("local/mock-edge", root=tmp_path / "lerobot_dataset")
    sample = dataset[0]

    assert dataset.num_episodes == 1
    assert dataset.num_frames == 3
    assert ACTION in sample
    assert OBS_STATE in sample
    assert REWARD in sample
    assert DONE in sample
    assert sample[ACTION].shape[0] == 3
    assert sample[OBS_STATE].shape[0] == 3
    assert sample[OBS_ENV_STATE].shape[0] == 3
    assert sample["task"] == "materialize-task"
    assert bool(sample[DONE].item()) is False
    assert float(sample[REWARD].item()) == 0.0


def test_materialized_lerobot_dataset_supports_one_step_training_smoke(tmp_path):
    _build_committed_episode(tmp_path)

    materializer = FilesystemEpisodeMaterializer(
        ingestion_root=tmp_path / "ingestion",
        output_root=tmp_path / "materialized",
    )
    materializer.materialize_to_lerobot_dataset(
        repo_id="local/mock-edge-train",
        dataset_root=tmp_path / "lerobot_dataset",
        fps=20,
    )

    result = run_training_smoke(
        dataset_root=str(tmp_path / "lerobot_dataset"),
        repo_id="local/mock-edge-train",
        output_dir=str(tmp_path / "train_out"),
    )

    assert result["policy_type"] == "act"
    assert result["dataset_num_episodes"] == 1
    assert result["dataset_num_frames"] == 3
    assert result["loss"] >= 0.0
