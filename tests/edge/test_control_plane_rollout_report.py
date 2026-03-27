#!/usr/bin/env python

from __future__ import annotations

import json

from lerobot.control_plane import ReleaseRegistry
from lerobot.edge.incidents import EdgeIncidentRecord, FilesystemIncidentSink
from lerobot.scripts.control_plane_rollout_report import (
    ControlPlaneRolloutReportConfig,
    generate_rollout_report,
)


def test_generate_rollout_report_writes_joined_snapshot(tmp_path):
    registry = ReleaseRegistry(tmp_path / "registry")
    registry.set_channel_target(channel="prod", target_artifact_id="artifact-prod-002")
    registry.assign_device(device_id="edge-01", channel="prod")

    sink = FilesystemIncidentSink(tmp_path / "incidents")
    sink.record(
        EdgeIncidentRecord(
            episode_id="episode-42",
            device_id="edge-01",
            channel="prod",
            policy_artifact_id="artifact-prod-002",
            sync_action="activated",
            rollback_action="rolled_back",
            watchdog_incident={"code": "model_crash", "action": "rollback_or_safe_stop", "message": "boom"},
        )
    )

    model_state_root = tmp_path / "edge-01-models"
    model_state_root.mkdir()
    with (model_state_root / "state.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "active_artifact_id": "artifact-prod-001",
                "pending_artifact_id": None,
                "previous_active_artifact_id": "artifact-prod-002",
            },
            handle,
        )

    output_path = generate_rollout_report(
        ControlPlaneRolloutReportConfig(
            registry_root=str(tmp_path / "registry"),
            incident_root=str(tmp_path / "incidents"),
            report_root=str(tmp_path / "reports"),
            device_state_roots={"edge-01": str(model_state_root)},
        )
    )

    with output_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["channels"][0]["channel"] == "prod"
    assert payload["channels"][0]["rollback_count"] == 1
    assert payload["devices"][0]["device_id"] == "edge-01"
    assert payload["devices"][0]["active_artifact_id"] == "artifact-prod-001"
