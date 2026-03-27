#!/usr/bin/env python

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .watchdog import WatchdogIncident


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class EdgeIncidentRecord:
    episode_id: str
    device_id: str | None
    channel: str | None
    policy_artifact_id: str
    sync_action: str
    rollback_action: str
    created_at: str = field(default_factory=_utcnow)
    watchdog_incident: dict | None = None

    @classmethod
    def from_watchdog(
        cls,
        *,
        episode_id: str,
        device_id: str | None,
        channel: str | None,
        policy_artifact_id: str,
        sync_action: str,
        rollback_action: str,
        watchdog_incident: WatchdogIncident | None,
    ) -> "EdgeIncidentRecord":
        return cls(
            episode_id=episode_id,
            device_id=device_id,
            channel=channel,
            policy_artifact_id=policy_artifact_id,
            sync_action=sync_action,
            rollback_action=rollback_action,
            watchdog_incident=None if watchdog_incident is None else asdict(watchdog_incident),
        )

    def to_dict(self) -> dict:
        return asdict(self)


class FilesystemIncidentSink:
    """Durable local sink for watchdog and rollback incidents."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def record(self, incident: EdgeIncidentRecord) -> Path:
        path = self.root / f"{incident.episode_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(incident.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def list_incidents(self) -> list[EdgeIncidentRecord]:
        incidents: list[EdgeIncidentRecord] = []
        for path in sorted(self.root.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                incidents.append(EdgeIncidentRecord(**json.load(handle)))
        return incidents
