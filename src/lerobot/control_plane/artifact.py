#!/usr/bin/env python

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


ARTIFACT_MANIFEST_SCHEMA_VERSION = "v1"


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def compute_sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class ArtifactManifest:
    """Minimal deployment artifact contract for edge-safe rollout."""

    artifact_id: str
    base_checkpoint: str
    runtime_abi_version: str
    observation_schema_version: str
    action_schema_version: str
    env_processor_digest: str
    action_processor_digest: str
    stats_digest: str
    policy_digest: str
    compatible_robot_types: list[str]
    compatible_camera_layouts: list[str]
    eval_summary: dict[str, float]
    created_at: str = field(default_factory=_utcnow)
    manifest_schema_version: str = ARTIFACT_MANIFEST_SCHEMA_VERSION
    policy_config: dict = field(default_factory=dict)
    env_processor_config: dict = field(default_factory=dict)
    action_processor_config: dict = field(default_factory=dict)
    training_lineage: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "artifact_id",
            "base_checkpoint",
            "runtime_abi_version",
            "observation_schema_version",
            "action_schema_version",
            "env_processor_digest",
            "action_processor_digest",
            "stats_digest",
            "policy_digest",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} cannot be empty.")
        if not self.compatible_robot_types:
            raise ValueError("compatible_robot_types cannot be empty.")
        if not self.compatible_camera_layouts:
            raise ValueError("compatible_camera_layouts cannot be empty.")

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return output_path

    @classmethod
    def from_json(cls, path: str | Path) -> "ArtifactManifest":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**payload)
