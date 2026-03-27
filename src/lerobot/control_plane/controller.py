#!/usr/bin/env python

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifact import ArtifactManifest
from .registry import ReleaseRegistry


@dataclass(frozen=True)
class ChannelPublishResult:
    channel: str
    artifact_id: str
    registry_root: str
    rollout_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReleaseController:
    """Minimal control-plane helper that publishes built artifacts into release channels."""

    def __init__(self, registry_root: str | Path, artifact_root: str | Path):
        self.registry = ReleaseRegistry(registry_root)
        self.artifact_root = Path(artifact_root)

    def publish_artifact_to_channel(
        self,
        *,
        channel: str,
        artifact_id: str,
        rollout_reason: str | None = None,
    ) -> ChannelPublishResult:
        artifact_dir = self.artifact_root / artifact_id
        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Artifact '{artifact_id}' is missing manifest.json under {self.artifact_root}.")

        manifest = ArtifactManifest.from_json(manifest_path)
        if manifest.artifact_id != artifact_id:
            raise ValueError(
                f"Artifact manifest id mismatch: requested={artifact_id}, manifest={manifest.artifact_id}"
            )

        self.registry.set_channel_target(
            channel=channel,
            target_artifact_id=artifact_id,
            rollout_reason=rollout_reason,
        )
        return ChannelPublishResult(
            channel=channel,
            artifact_id=artifact_id,
            registry_root=str(self.registry.layout.root),
            rollout_reason=rollout_reason,
        )
