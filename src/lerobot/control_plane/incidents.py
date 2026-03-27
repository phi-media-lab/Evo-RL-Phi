#!/usr/bin/env python

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from lerobot.edge.incidents import EdgeIncidentRecord, FilesystemIncidentSink

from .registry import ReleaseRegistry


@dataclass(frozen=True)
class ChannelIncidentSummary:
    channel: str
    incident_count: int
    rollback_count: int
    latest_episode_id: str | None = None
    latest_policy_artifact_id: str | None = None
    latest_incident_code: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DeviceIncidentSummary:
    device_id: str
    channel: str
    incident_count: int
    rollback_count: int
    latest_episode_id: str | None = None
    latest_policy_artifact_id: str | None = None
    latest_incident_code: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class IncidentAggregationReport:
    channels: list[ChannelIncidentSummary] = field(default_factory=list)
    devices: list[DeviceIncidentSummary] = field(default_factory=list)


class IncidentAggregator:
    """Aggregates local incident records into channel- and device-level rollout summaries."""

    def __init__(self, incident_root: str | Path, registry: ReleaseRegistry):
        self.sink = FilesystemIncidentSink(incident_root)
        self.registry = registry

    def build_report(self) -> IncidentAggregationReport:
        incidents = self.sink.list_incidents()
        return IncidentAggregationReport(
            channels=self._build_channel_summaries(incidents),
            devices=self._build_device_summaries(incidents),
        )

    def _build_channel_summaries(self, incidents: list[EdgeIncidentRecord]) -> list[ChannelIncidentSummary]:
        summaries: dict[str, ChannelIncidentSummary] = {}
        for incident in incidents:
            channel = incident.channel or self._resolve_channel_for_device(incident.device_id) or "unassigned"
            previous = summaries.get(channel)
            incident_count = 1 if incident.watchdog_incident is not None else 0
            rollback_count = 1 if incident.rollback_action == "rolled_back" else 0
            latest_incident_code = None
            if incident.watchdog_incident is not None:
                latest_incident_code = incident.watchdog_incident.get("code")
            if previous is None:
                summaries[channel] = ChannelIncidentSummary(
                    channel=channel,
                    incident_count=incident_count,
                    rollback_count=rollback_count,
                    latest_episode_id=incident.episode_id,
                    latest_policy_artifact_id=incident.policy_artifact_id,
                    latest_incident_code=latest_incident_code,
                )
                continue
            summaries[channel] = ChannelIncidentSummary(
                channel=channel,
                incident_count=previous.incident_count + incident_count,
                rollback_count=previous.rollback_count + rollback_count,
                latest_episode_id=incident.episode_id,
                latest_policy_artifact_id=incident.policy_artifact_id,
                latest_incident_code=latest_incident_code,
            )
        return sorted(summaries.values(), key=lambda item: item.channel)

    def _build_device_summaries(self, incidents: list[EdgeIncidentRecord]) -> list[DeviceIncidentSummary]:
        summaries: dict[str, DeviceIncidentSummary] = {}
        for incident in incidents:
            device_id = incident.device_id or "unknown-device"
            channel = incident.channel or self._resolve_channel_for_device(incident.device_id) or "unassigned"
            previous = summaries.get(device_id)
            incident_count = 1 if incident.watchdog_incident is not None else 0
            rollback_count = 1 if incident.rollback_action == "rolled_back" else 0
            latest_incident_code = None
            if incident.watchdog_incident is not None:
                latest_incident_code = incident.watchdog_incident.get("code")
            if previous is None:
                summaries[device_id] = DeviceIncidentSummary(
                    device_id=device_id,
                    channel=channel,
                    incident_count=incident_count,
                    rollback_count=rollback_count,
                    latest_episode_id=incident.episode_id,
                    latest_policy_artifact_id=incident.policy_artifact_id,
                    latest_incident_code=latest_incident_code,
                )
                continue
            summaries[device_id] = DeviceIncidentSummary(
                device_id=device_id,
                channel=channel,
                incident_count=previous.incident_count + incident_count,
                rollback_count=previous.rollback_count + rollback_count,
                latest_episode_id=incident.episode_id,
                latest_policy_artifact_id=incident.policy_artifact_id,
                latest_incident_code=latest_incident_code,
            )
        return sorted(summaries.values(), key=lambda item: item.device_id)

    def _resolve_channel_for_device(self, device_id: str | None) -> str | None:
        if device_id is None:
            return None
        assignment = self.registry.get_device_assignment(device_id)
        if assignment is None:
            return None
        return assignment.channel
