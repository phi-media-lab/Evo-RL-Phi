#!/usr/bin/env python

from __future__ import annotations

from lerobot.control_plane import IncidentAggregator, ReleaseRegistry
from lerobot.edge.incidents import EdgeIncidentRecord, FilesystemIncidentSink


def test_incident_aggregator_builds_channel_and_device_summaries(tmp_path):
    registry = ReleaseRegistry(tmp_path / "registry")
    registry.set_channel_target(channel="prod", target_artifact_id="artifact-prod-002")
    registry.assign_device(device_id="edge-01", channel="prod")
    registry.assign_device(device_id="edge-02", channel="prod")

    sink = FilesystemIncidentSink(tmp_path / "incidents")
    sink.record(
        EdgeIncidentRecord(
            episode_id="episode-1",
            device_id="edge-01",
            channel="prod",
            policy_artifact_id="artifact-prod-002",
            sync_action="activated",
            rollback_action="rolled_back",
            watchdog_incident={"code": "model_crash", "action": "rollback_or_safe_stop", "message": "boom"},
        )
    )
    sink.record(
        EdgeIncidentRecord(
            episode_id="episode-2",
            device_id="edge-02",
            channel="prod",
            policy_artifact_id="artifact-prod-002",
            sync_action="noop",
            rollback_action="none",
            watchdog_incident=None,
        )
    )

    aggregator = IncidentAggregator(tmp_path / "incidents", registry)
    report = aggregator.build_report()

    assert len(report.channels) == 1
    assert report.channels[0].channel == "prod"
    assert report.channels[0].incident_count == 1
    assert report.channels[0].rollback_count == 1
    assert report.channels[0].latest_episode_id == "episode-2"

    assert len(report.devices) == 2
    assert report.devices[0].device_id == "edge-01"
    assert report.devices[0].incident_count == 1
    assert report.devices[0].rollback_count == 1
    assert report.devices[1].device_id == "edge-02"
    assert report.devices[1].incident_count == 0
