#!/usr/bin/env python

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum


class ArtifactStatus(str, Enum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    APPROVED = "approved"
    DEPRECATED = "deprecated"


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class ReleaseChannelState:
    """Current target artifact for a deployment channel."""

    channel: str
    target_artifact_id: str
    updated_at: str = field(default_factory=_utcnow)
    rollout_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.channel:
            raise ValueError("channel cannot be empty.")
        if not self.target_artifact_id:
            raise ValueError("target_artifact_id cannot be empty.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DeviceChannelAssignment:
    """Binds a robot or edge node to a release channel."""

    device_id: str
    channel: str
    assigned_at: str = field(default_factory=_utcnow)
    robot_id: str | None = None
    task_id: str | None = None

    def __post_init__(self) -> None:
        if not self.device_id:
            raise ValueError("device_id cannot be empty.")
        if not self.channel:
            raise ValueError("channel cannot be empty.")

    def to_dict(self) -> dict:
        return asdict(self)
