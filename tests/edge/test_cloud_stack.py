#!/usr/bin/env python

from __future__ import annotations

import json
import threading
from urllib import request as urllib_request

from lerobot.scripts.cloud_stack import CloudServiceGroup, run_cloud_stack
from tests.edge.test_materializer import _build_committed_episode


def test_cloud_service_group_exposes_http_endpoints(tmp_path):
    services = CloudServiceGroup(
        host="127.0.0.1",
        ingestion_port=0,
        materializer_port=0,
        ingestion_root=str(tmp_path / "ingestion"),
        materialized_root=str(tmp_path / "materialized"),
        dataset_root=str(tmp_path / "dataset"),
    )
    try:
        endpoints = services.start()
    finally:
        services.close()

    assert endpoints.ingestion_base_url.startswith("http://127.0.0.1:")
    assert endpoints.materializer_base_url.startswith("http://127.0.0.1:")


def test_run_cloud_stack_releases_from_ingestion_snapshot(tmp_path):
    _build_committed_episode(tmp_path)

    output = run_cloud_stack(
        host="127.0.0.1",
        ingestion_port=0,
        materializer_port=0,
        status_port=None,
        ingestion_root=str(tmp_path / "ingestion"),
        materialized_root=str(tmp_path / "materialized"),
        dataset_root=str(tmp_path / "dataset"),
        train_output_root=str(tmp_path / "train_runs"),
        artifact_output_root=str(tmp_path / "artifacts"),
        registry_root=str(tmp_path / "registry"),
        state_root=str(tmp_path / "controller_state"),
        runtime_root=str(tmp_path / "controller_runtime"),
        incident_root=str(tmp_path / "incidents"),
        report_root=str(tmp_path / "reports"),
        channel="staging",
        artifact_prefix="artifact-cloud-stack",
        compatible_robot_types=["mock_robot"],
        compatible_camera_layouts=["single_arm_mock"],
        rollout_reason="cloud stack release",
        poll_interval_s=0.0,
        max_iterations=1,
    )

    latest_payload = json.loads((tmp_path / "controller_runtime" / "latest.json").read_text(encoding="utf-8"))
    report_payload = json.loads((tmp_path / "reports" / "rollout_status.json").read_text(encoding="utf-8"))
    stack_status = json.loads((tmp_path / "controller_runtime" / "cloud_stack_status.json").read_text(encoding="utf-8"))
    staging_channel = next(channel for channel in report_payload["channels"] if channel["channel"] == "staging")

    assert "auto-release-daemon iterations=1" in output
    assert latest_payload["action"] == "released"
    assert staging_channel["target_artifact_id"] == latest_payload["artifact_id"]
    assert stack_status["phase"] == "completed"
    assert stack_status["latest_action"] == "released"
    assert stack_status["latest_artifact_id"] == latest_payload["artifact_id"]


def test_cloud_stack_status_server_reports_health_and_status(tmp_path):
    _build_committed_episode(tmp_path)

    container: dict[str, str] = {}

    def _run_stack() -> None:
        container["output"] = run_cloud_stack(
            host="127.0.0.1",
            ingestion_port=0,
            materializer_port=0,
            status_port=0,
            ingestion_root=str(tmp_path / "ingestion"),
            materialized_root=str(tmp_path / "materialized"),
            dataset_root=str(tmp_path / "dataset"),
            train_output_root=str(tmp_path / "train_runs"),
            artifact_output_root=str(tmp_path / "artifacts"),
            registry_root=str(tmp_path / "registry"),
            state_root=str(tmp_path / "controller_state"),
            runtime_root=str(tmp_path / "controller_runtime"),
            incident_root=str(tmp_path / "incidents"),
            report_root=str(tmp_path / "reports"),
            channel="staging",
            artifact_prefix="artifact-cloud-stack",
            compatible_robot_types=["mock_robot"],
            compatible_camera_layouts=["single_arm_mock"],
            rollout_reason="cloud stack release",
            poll_interval_s=0.2,
            max_iterations=1,
        )

    thread = threading.Thread(target=_run_stack, daemon=True)
    thread.start()

    status_path = tmp_path / "controller_runtime" / "cloud_stack_status.json"
    for _ in range(200):
        if status_path.exists():
            break
        thread.join(timeout=0.05)

    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    status_base_url = status_payload["endpoints"]["status_base_url"]

    with urllib_request.urlopen(f"{status_base_url}/healthz") as response:
        health_payload = json.loads(response.read().decode("utf-8"))
    with urllib_request.urlopen(f"{status_base_url}/status") as response:
        runtime_payload = json.loads(response.read().decode("utf-8"))

    thread.join(timeout=30)
    assert not thread.is_alive()
    assert health_payload["ok"] is True
    assert runtime_payload["phase"] in {"running", "completed"}
    assert runtime_payload["endpoints"]["ingestion_base_url"].startswith("http://127.0.0.1:")
