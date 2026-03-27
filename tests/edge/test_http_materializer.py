#!/usr/bin/env python

from __future__ import annotations

import threading

from lerobot.cloud.materializer import HTTPMaterializerClient, MaterializerHTTPServer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tests.edge.test_materializer import _build_committed_episode


def _start_materializer_server(tmp_path):
    server = MaterializerHTTPServer(
        ("127.0.0.1", 0),
        ingestion_root=tmp_path / "ingestion",
        output_root=tmp_path / "materialized",
        dataset_root=tmp_path / "lerobot_dataset",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_http_materializer_builds_json_manifest(tmp_path):
    result = _build_committed_episode(tmp_path)
    server, thread = _start_materializer_server(tmp_path)
    client = HTTPMaterializerClient(f"http://127.0.0.1:{server.server_port}")
    try:
        receipt = client.materialize()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    manifest = receipt["manifest"]
    assert receipt["status"] == "materialized"
    assert manifest["episode_count"] == 1
    assert manifest["total_step_count"] == 3
    assert manifest["episodes"][0]["episode_id"] == result.episode_id
    assert (tmp_path / "materialized" / "manifest.json").exists()


def test_http_materializer_exports_lerobot_dataset(tmp_path):
    _build_committed_episode(tmp_path)
    server, thread = _start_materializer_server(tmp_path)
    client = HTTPMaterializerClient(f"http://127.0.0.1:{server.server_port}")
    try:
        receipt = client.materialize(
            repo_id="local/http-materialized",
            fps=20,
            export_lerobot_dataset=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    manifest = receipt["manifest"]
    dataset = LeRobotDataset("local/http-materialized", root=tmp_path / "lerobot_dataset")

    assert manifest["lerobot_repo_id"] == "local/http-materialized"
    assert manifest["lerobot_root"] == str(tmp_path / "lerobot_dataset")
    assert dataset.num_episodes == 1
    assert dataset.num_frames == 3
