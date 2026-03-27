#!/usr/bin/env python

from __future__ import annotations

import json
import threading

from lerobot.cloud.ingestion import (
    EpisodeChunk,
    EpisodeCommitRequest,
    EpisodeIngestionHTTPServer,
    FilesystemEpisodeIngestionStore,
    HTTPEpisodeIngestionClient,
)
from lerobot.edge.mock_runtime import StaticActionRuntime, StaticActionRuntimeConfig
from lerobot.edge.runner import EdgeRobotRunner
from lerobot.edge.spool import EdgeEpisodeSpool
from lerobot.edge.uploader import EdgeEpisodeUploader, EdgeUploaderConfig
from lerobot.robots.utils import make_robot_from_config
from tests.mocks.mock_robot import MockRobotConfig


def _start_ingestion_server(tmp_path):
    server = EpisodeIngestionHTTPServer(("127.0.0.1", 0), FilesystemEpisodeIngestionStore(tmp_path / "ingestion"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_http_ingestion_client_roundtrip(tmp_path):
    server, thread = _start_ingestion_server(tmp_path)
    client = HTTPEpisodeIngestionClient(f"http://127.0.0.1:{server.server_port}")
    try:
        chunk_receipt = client.upload_chunk(
            EpisodeChunk(
                episode_id="episode-http-001",
                chunk_index=0,
                payload_digest="sha256:test",
                byte_size=12,
                step_count=1,
            ),
            '{"step_idx": 0}\n',
        )
        receipt = client.commit_episode(
            EpisodeCommitRequest(
                episode_id="episode-http-001",
                expected_chunk_count=1,
                final_step_idx=0,
            ),
            episode_meta={"episode_id": "episode-http-001", "task_id": "demo", "robot_id": "robot-01"},
            episode_summary={"step_count": 1},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert chunk_receipt["episode_id"] == "episode-http-001"
    assert receipt["status"] == "committed"
    assert (tmp_path / "ingestion" / "episode-http-001" / "commit.json").exists()


def test_edge_uploader_supports_http_ingestion_sink(tmp_path):
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

    robot.connect()
    try:
        run_result = EdgeRobotRunner(
            robot=robot,
            runtime=runtime,
            spool=spool,
            policy_artifact_id="http-upload-policy",
            processor_bundle_id="http-upload-processor",
            task_id="http-upload-task",
        ).run_episode(max_steps=3, fps=20)
    finally:
        robot.disconnect()

    server, thread = _start_ingestion_server(tmp_path)
    try:
        receipt = EdgeEpisodeUploader(
            spool=spool,
            sink=HTTPEpisodeIngestionClient(f"http://127.0.0.1:{server.server_port}"),
            config=EdgeUploaderConfig(steps_per_chunk=2),
        ).upload_episode(run_result.episode_id)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    uploaded_dir = tmp_path / "spool" / "uploaded" / run_result.episode_id
    commit_path = tmp_path / "ingestion" / run_result.episode_id / "commit.json"
    receipt_path = uploaded_dir / "upload_receipt.json"

    assert receipt["status"] == "committed"
    assert commit_path.exists()
    assert receipt_path.exists()

    with receipt_path.open("r", encoding="utf-8") as handle:
        saved_receipt = json.load(handle)
    assert saved_receipt["episode_id"] == run_result.episode_id
