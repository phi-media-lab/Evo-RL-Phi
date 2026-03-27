#!/usr/bin/env python

from __future__ import annotations

import json

from lerobot.control_plane import IncidentAggregator, ReleaseRegistry, RolloutStatusBuilder, RolloutStatusStore
from lerobot.edge.incidents import EdgeIncidentRecord, FilesystemIncidentSink


def test_rollout_status_builder_joins_registry_incidents_and_edge_state(tmp_path):
    registry = ReleaseRegistry(tmp_path / "registry")
    registry.set_channel_target(channel="prod", target_artifact_id="artifact-prod-002")
    registry.assign_device(device_id="edge-01", channel="prod")

    incident_sink = FilesystemIncidentSink(tmp_path / "incidents")
    incident_sink.record(
        EdgeIncidentRecord(
            episode_id="episode-9",
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
                "previous_active_artifact_id": "artifact-prod-002",
                "pending_artifact_id": None,
            },
            handle,
        )

    builder = RolloutStatusBuilder(
        registry=registry,
        incident_aggregator=IncidentAggregator(tmp_path / "incidents", registry),
        device_state_roots={"edge-01": model_state_root},
    )
    report = builder.build_report()

    assert len(report.channels) == 1
    assert report.channels[0].channel == "prod"
    assert report.channels[0].target_artifact_id == "artifact-prod-002"
    assert report.channels[0].assigned_device_count == 1
    assert report.channels[0].rollback_count == 1

    assert len(report.devices) == 1
    assert report.devices[0].device_id == "edge-01"
    assert report.devices[0].target_artifact_id == "artifact-prod-002"
    assert report.devices[0].active_artifact_id == "artifact-prod-001"
    assert report.devices[0].latest_incident_code == "model_crash"


def test_rollout_status_store_writes_json_report(tmp_path):
    registry = ReleaseRegistry(tmp_path / "registry")
    aggregator = IncidentAggregator(tmp_path / "incidents", registry)
    builder = RolloutStatusBuilder(registry=registry, incident_aggregator=aggregator)
    store = RolloutStatusStore(tmp_path / "reports")

    report = builder.build_report()
    output_path = store.write_report(report)

    with output_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload == {"channels": [], "devices": []}
