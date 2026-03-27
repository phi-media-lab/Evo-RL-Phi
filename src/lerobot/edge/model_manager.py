#!/usr/bin/env python

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lerobot.control_plane.artifact import ArtifactManifest
from lerobot.control_plane.registry import ReleaseRegistry

from .runtime import LocalPolicyRuntime, LocalPolicyRuntimeConfig


@dataclass(frozen=True)
class EdgeCompatibilityContext:
    """Static compatibility inputs for deciding whether an artifact can run on this edge node."""

    robot_type: str
    camera_layout: str
    runtime_abi_version: str = "v1"
    observation_schema_version: str = "v1"
    action_schema_version: str = "v1"


@dataclass(frozen=True)
class ManagedArtifact:
    """Local artifact with resolved runtime paths."""

    root: Path
    manifest: ArtifactManifest
    policy_path: Path
    preprocessor_config_filename: str | None = None
    postprocessor_config_filename: str | None = None


@dataclass
class EdgeModelManagerState:
    active_artifact_id: str | None = None
    previous_active_artifact_id: str | None = None
    pending_artifact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EdgeDeploymentSyncResult:
    action: str
    channel: str
    target_artifact_id: str
    changed: bool
    runtime: LocalPolicyRuntime | None = None


class EdgeModelManager:
    """Minimal local model manager for artifact validation, staging, activation, and rollback."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state = self._load_state()

    def discover_artifact_dirs(self) -> list[Path]:
        return sorted(path for path in self.root.iterdir() if path.is_dir() and (path / "manifest.json").exists())

    def load_artifact(self, artifact_dir: str | Path) -> ManagedArtifact:
        artifact_root = Path(artifact_dir)
        manifest = ArtifactManifest.from_json(artifact_root / "manifest.json")

        policy_path_value = manifest.metadata.get("policy_path", ".")
        policy_path = artifact_root / policy_path_value
        preprocessor_config_filename = manifest.metadata.get("preprocessor_config_filename")
        postprocessor_config_filename = manifest.metadata.get("postprocessor_config_filename")
        return ManagedArtifact(
            root=artifact_root,
            manifest=manifest,
            policy_path=policy_path,
            preprocessor_config_filename=preprocessor_config_filename,
            postprocessor_config_filename=postprocessor_config_filename,
        )

    def load_artifact_by_id(self, artifact_id: str) -> ManagedArtifact:
        artifact_root = self.root / artifact_id
        if not artifact_root.exists():
            raise FileNotFoundError(f"Artifact '{artifact_id}' does not exist under {self.root}.")
        return self.load_artifact(artifact_root)

    def validate_artifact(self, artifact: ManagedArtifact, context: EdgeCompatibilityContext) -> list[str]:
        errors: list[str] = []
        manifest = artifact.manifest

        if manifest.runtime_abi_version != context.runtime_abi_version:
            errors.append(
                f"runtime ABI mismatch: artifact={manifest.runtime_abi_version}, edge={context.runtime_abi_version}"
            )
        if manifest.observation_schema_version != context.observation_schema_version:
            errors.append(
                "observation schema mismatch: "
                f"artifact={manifest.observation_schema_version}, edge={context.observation_schema_version}"
            )
        if manifest.action_schema_version != context.action_schema_version:
            errors.append(
                f"action schema mismatch: artifact={manifest.action_schema_version}, edge={context.action_schema_version}"
            )
        if context.robot_type not in manifest.compatible_robot_types:
            errors.append(f"robot type '{context.robot_type}' not allowed by artifact.")
        if context.camera_layout not in manifest.compatible_camera_layouts:
            errors.append(f"camera layout '{context.camera_layout}' not allowed by artifact.")
        if not artifact.policy_path.exists():
            errors.append(f"policy path does not exist: {artifact.policy_path}")

        return errors

    def stage_artifact(self, artifact: ManagedArtifact, context: EdgeCompatibilityContext) -> None:
        errors = self.validate_artifact(artifact, context)
        if errors:
            raise ValueError("Artifact compatibility validation failed: " + "; ".join(errors))
        self.state.pending_artifact_id = artifact.manifest.artifact_id
        self._write_state()

    def activate_artifact(
        self,
        artifact: ManagedArtifact,
        *,
        runtime_overrides: dict[str, Any] | None = None,
        warmup_observation: dict[str, Any] | None = None,
    ) -> LocalPolicyRuntime:
        runtime_config = self.build_runtime_config(artifact, overrides=runtime_overrides)
        runtime = LocalPolicyRuntime(runtime_config)
        if warmup_observation is not None:
            runtime.warmup(warmup_observation)

        if self.state.active_artifact_id != artifact.manifest.artifact_id:
            self.state.previous_active_artifact_id = self.state.active_artifact_id
        self.state.active_artifact_id = artifact.manifest.artifact_id
        if self.state.pending_artifact_id == artifact.manifest.artifact_id:
            self.state.pending_artifact_id = None
        self._write_state()
        return runtime

    def rollback_to_active(
        self,
        artifact: ManagedArtifact,
        *,
        runtime_overrides: dict[str, Any] | None = None,
        warmup_observation: dict[str, Any] | None = None,
    ) -> LocalPolicyRuntime:
        self.state.pending_artifact_id = None
        self._write_state()
        return self.activate_artifact(
            artifact,
            runtime_overrides=runtime_overrides,
            warmup_observation=warmup_observation,
        )

    def build_runtime_config(
        self,
        artifact: ManagedArtifact,
        overrides: dict[str, Any] | None = None,
    ) -> LocalPolicyRuntimeConfig:
        policy_type = artifact.manifest.policy_config.get("type")
        if policy_type is None:
            raise ValueError("Artifact manifest missing policy_config.type")

        runtime_kwargs: dict[str, Any] = {
            "policy_type": policy_type,
            "pretrained_name_or_path": str(artifact.policy_path),
            "task": artifact.manifest.metadata.get("task", ""),
            "robot_type": artifact.manifest.metadata.get("robot_type", ""),
        }
        if artifact.preprocessor_config_filename is not None:
            runtime_kwargs["preprocessor_config_filename"] = artifact.preprocessor_config_filename
        if artifact.postprocessor_config_filename is not None:
            runtime_kwargs["postprocessor_config_filename"] = artifact.postprocessor_config_filename
        runtime_kwargs.update(artifact.manifest.metadata.get("runtime", {}))
        if overrides is not None:
            runtime_kwargs.update(overrides)
        return LocalPolicyRuntimeConfig(**runtime_kwargs)

    def get_active_artifact_id(self) -> str | None:
        return self.state.active_artifact_id

    def get_pending_artifact_id(self) -> str | None:
        return self.state.pending_artifact_id

    def get_previous_active_artifact_id(self) -> str | None:
        return self.state.previous_active_artifact_id

    def resolve_channel_artifact(
        self,
        registry: ReleaseRegistry,
        *,
        device_id: str,
        fallback_channel: str | None = None,
    ) -> ManagedArtifact:
        channel_state = registry.resolve_target_for_device(device_id, fallback_channel=fallback_channel)
        return self.load_artifact_by_id(channel_state.target_artifact_id)

    def sync_to_registry_target(
        self,
        registry: ReleaseRegistry,
        *,
        device_id: str,
        context: EdgeCompatibilityContext,
        fallback_channel: str | None = None,
        activate: bool = False,
        runtime_overrides: dict[str, Any] | None = None,
        warmup_observation: dict[str, Any] | None = None,
    ) -> EdgeDeploymentSyncResult:
        channel_state = registry.resolve_target_for_device(device_id, fallback_channel=fallback_channel)
        target_artifact_id = channel_state.target_artifact_id
        if self.state.active_artifact_id == target_artifact_id:
            return EdgeDeploymentSyncResult(
                action="noop",
                channel=channel_state.channel,
                target_artifact_id=target_artifact_id,
                changed=False,
            )

        artifact = self.load_artifact_by_id(target_artifact_id)
        self.stage_artifact(artifact, context)
        if not activate:
            return EdgeDeploymentSyncResult(
                action="staged",
                channel=channel_state.channel,
                target_artifact_id=target_artifact_id,
                changed=True,
            )

        runtime = self.activate_artifact(
            artifact,
            runtime_overrides=runtime_overrides,
            warmup_observation=warmup_observation,
        )
        return EdgeDeploymentSyncResult(
            action="activated",
            channel=channel_state.channel,
            target_artifact_id=target_artifact_id,
            changed=True,
            runtime=runtime,
        )

    def rollback_to_previous_active(
        self,
        *,
        runtime_overrides: dict[str, Any] | None = None,
        warmup_observation: dict[str, Any] | None = None,
    ) -> LocalPolicyRuntime:
        previous_active_artifact_id = self.state.previous_active_artifact_id
        if previous_active_artifact_id is None:
            raise ValueError("No previous active artifact available for rollback.")
        artifact = self.load_artifact_by_id(previous_active_artifact_id)
        runtime = self.activate_artifact(
            artifact,
            runtime_overrides=runtime_overrides,
            warmup_observation=warmup_observation,
        )
        self.state.pending_artifact_id = None
        self._write_state()
        return runtime

    def _load_state(self) -> EdgeModelManagerState:
        if not self.state_path.exists():
            return EdgeModelManagerState()
        with self.state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return EdgeModelManagerState(**payload)

    def _write_state(self) -> None:
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(self.state.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
