#!/usr/bin/env python

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .incidents import IncidentAggregator
from .registry import ReleaseRegistry


@dataclass(frozen=True)
class DeviceRolloutStatus:
    device_id: str
    channel: str
    target_artifact_id: str | None
    active_artifact_id: str | None
    pending_artifact_id: str | None
    previous_active_artifact_id: str | None
    incident_count: int = 0
    rollback_count: int = 0
    latest_incident_code: str | None = None
    latest_episode_id: str | None = None
    latest_policy_artifact_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ChannelRolloutStatus:
    channel: str
    target_artifact_id: str | None
    assigned_device_count: int
    incident_count: int = 0
    rollback_count: int = 0
    latest_incident_code: str | None = None
    latest_episode_id: str | None = None
    latest_policy_artifact_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RolloutStatusReport:
    channels: list[ChannelRolloutStatus] = field(default_factory=list)
    devices: list[DeviceRolloutStatus] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "channels": [item.to_dict() for item in self.channels],
            "devices": [item.to_dict() for item in self.devices],
        }


class RolloutStatusBuilder:
    """Builds a control-plane snapshot by joining registry, incidents, and edge model states."""

    def __init__(
        self,
        *,
        registry: ReleaseRegistry,
        incident_aggregator: IncidentAggregator,
        device_state_roots: dict[str, str | Path] | None = None,
    ):
        self.registry = registry
        self.incident_aggregator = incident_aggregator
        self.device_state_roots = {
            device_id: Path(root) for device_id, root in (device_state_roots or {}).items()
        }

    def build_report(self) -> RolloutStatusReport:
        incident_report = self.incident_aggregator.build_report()
        device_incidents = {item.device_id: item for item in incident_report.devices}
        channel_incidents = {item.channel: item for item in incident_report.channels}

        device_statuses: list[DeviceRolloutStatus] = []
        assignments = self.registry.list_device_assignments()
        for assignment in assignments:
            channel_target = self.registry.get_channel_target(assignment.channel)
            state = self._read_device_state(assignment.device_id)
            incident = device_incidents.get(assignment.device_id)
            device_statuses.append(
                DeviceRolloutStatus(
                    device_id=assignment.device_id,
                    channel=assignment.channel,
                    target_artifact_id=None if channel_target is None else channel_target.target_artifact_id,
                    active_artifact_id=state.get("active_artifact_id"),
                    pending_artifact_id=state.get("pending_artifact_id"),
                    previous_active_artifact_id=state.get("previous_active_artifact_id"),
                    incident_count=0 if incident is None else incident.incident_count,
                    rollback_count=0 if incident is None else incident.rollback_count,
                    latest_incident_code=None if incident is None else incident.latest_incident_code,
                    latest_episode_id=None if incident is None else incident.latest_episode_id,
                    latest_policy_artifact_id=None if incident is None else incident.latest_policy_artifact_id,
                )
            )

        assigned_counts: dict[str, int] = {}
        for assignment in assignments:
            assigned_counts[assignment.channel] = assigned_counts.get(assignment.channel, 0) + 1

        channel_statuses: list[ChannelRolloutStatus] = []
        for channel_state in self.registry.list_channel_targets():
            incident = channel_incidents.get(channel_state.channel)
            channel_statuses.append(
                ChannelRolloutStatus(
                    channel=channel_state.channel,
                    target_artifact_id=channel_state.target_artifact_id,
                    assigned_device_count=assigned_counts.get(channel_state.channel, 0),
                    incident_count=0 if incident is None else incident.incident_count,
                    rollback_count=0 if incident is None else incident.rollback_count,
                    latest_incident_code=None if incident is None else incident.latest_incident_code,
                    latest_episode_id=None if incident is None else incident.latest_episode_id,
                    latest_policy_artifact_id=None if incident is None else incident.latest_policy_artifact_id,
                )
            )

        return RolloutStatusReport(
            channels=sorted(channel_statuses, key=lambda item: item.channel),
            devices=sorted(device_statuses, key=lambda item: item.device_id),
        )

    def _read_device_state(self, device_id: str) -> dict:
        root = self.device_state_roots.get(device_id)
        if root is None:
            return {}
        state_path = root / "state.json"
        if not state_path.exists():
            return {}
        with state_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


class RolloutStatusStore:
    """Writes control-plane rollout snapshots as durable JSON reports."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_report(self, report: RolloutStatusReport, filename: str = "rollout_status.json") -> Path:
        path = self.root / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path
