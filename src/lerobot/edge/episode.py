#!/usr/bin/env python

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class EdgeEpisodeStepRecord:
    """Canonical step record for local recording and upload."""

    episode_id: str
    step_idx: int
    obs_ts: str
    action_ts: str
    teleop_override: bool
    reward_raw: float
    done_local: bool
    done_train: bool
    reward_relabel_version: str | None = None
    reward_relabel: float | None = None
    observation: dict | None = None
    action: dict | list | float | int | None = None
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id cannot be empty.")
        if self.step_idx < 0:
            raise ValueError("step_idx must be >= 0.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EdgeEpisodeRecord:
    """Canonical episode record for local recording and artifact traceability."""

    robot_id: str
    policy_artifact_id: str
    processor_bundle_id: str
    task_id: str
    episode_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: str = field(default_factory=lambda: _utcnow().isoformat())
    runtime_abi_version: str = "v1"
    observation_schema_version: str = "v1"
    action_schema_version: str = "v1"
    channel: str | None = None
    robot_type: str | None = None
    camera_layout: str | None = None
    robot_config_hash: str | None = None
    processor_config_hash: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("robot_id", "policy_artifact_id", "processor_bundle_id", "task_id", "episode_id"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} cannot be empty.")

    def to_dict(self) -> dict:
        return asdict(self)
