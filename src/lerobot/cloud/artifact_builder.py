#!/usr/bin/env python

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lerobot.control_plane.artifact import ArtifactManifest, compute_sha256, compute_sha256_bytes


def compute_tree_digest(root: str | Path) -> str:
    root = Path(root)
    payload_parts: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel_path = path.relative_to(root).as_posix()
        payload_parts.append(f"{rel_path}:{compute_sha256(path)}")
    return compute_sha256_bytes("\n".join(payload_parts).encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class ArtifactBuildRequest:
    artifact_id: str
    base_checkpoint: str
    policy_dir: str | Path
    output_root: str | Path
    compatible_robot_types: list[str]
    compatible_camera_layouts: list[str]
    runtime_abi_version: str = "v1"
    observation_schema_version: str = "v1"
    action_schema_version: str = "v1"
    env_processor_config: dict[str, Any] = field(default_factory=dict)
    action_processor_config: dict[str, Any] = field(default_factory=dict)
    eval_summary: dict[str, float] = field(default_factory=dict)
    training_lineage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    stats_path: str | Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy_dir"] = str(self.policy_dir)
        payload["output_root"] = str(self.output_root)
        if self.stats_path is not None:
            payload["stats_path"] = str(self.stats_path)
        return payload


@dataclass(frozen=True)
class OpenPIArtifactBuildRequest:
    artifact_id: str
    base_checkpoint: str
    output_root: str | Path
    compatible_robot_types: list[str]
    compatible_camera_layouts: list[str]
    policy_ref: str
    runtime_backend: str
    action_horizon: int
    action_dim: int
    observation_contract: dict[str, Any]
    model_output_key: str = ""
    output_transform: str = ""
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    runtime_assets_dir: str | Path | None = None
    runtime_abi_version: str = "v1"
    observation_schema_version: str = "v1"
    action_schema_version: str = "v1"
    env_processor_config: dict[str, Any] = field(default_factory=dict)
    action_processor_config: dict[str, Any] = field(default_factory=dict)
    eval_summary: dict[str, float] = field(default_factory=dict)
    training_lineage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        if self.runtime_assets_dir is not None:
            payload["runtime_assets_dir"] = str(self.runtime_assets_dir)
        return payload


class FilesystemArtifactBuilder:
    """Builds edge-compatible deployment artifacts from local pretrained policy directories."""

    def build(self, request: ArtifactBuildRequest) -> Path:
        policy_dir = Path(request.policy_dir)
        if not policy_dir.exists():
            raise FileNotFoundError(f"Policy directory does not exist: {policy_dir}")

        output_root = Path(request.output_root)
        artifact_root = output_root / request.artifact_id
        if artifact_root.exists():
            shutil.rmtree(artifact_root)
        artifact_root.mkdir(parents=True, exist_ok=True)

        copied_policy_dir = artifact_root / "policy"
        shutil.copytree(policy_dir, copied_policy_dir)

        stats_dest = artifact_root / "stats.json"
        stats_digest = self._write_stats(request.stats_path, stats_dest)

        env_processor_path = artifact_root / "env_processor_config.json"
        action_processor_path = artifact_root / "action_processor_config.json"
        self._write_json(env_processor_path, request.env_processor_config)
        self._write_json(action_processor_path, request.action_processor_config)

        policy_config = _read_json(copied_policy_dir / "config.json")
        manifest = ArtifactManifest(
            artifact_id=request.artifact_id,
            base_checkpoint=request.base_checkpoint,
            runtime_abi_version=request.runtime_abi_version,
            observation_schema_version=request.observation_schema_version,
            action_schema_version=request.action_schema_version,
            env_processor_digest=compute_sha256(env_processor_path),
            action_processor_digest=compute_sha256(action_processor_path),
            stats_digest=stats_digest,
            policy_digest=compute_tree_digest(copied_policy_dir),
            compatible_robot_types=request.compatible_robot_types,
            compatible_camera_layouts=request.compatible_camera_layouts,
            eval_summary=request.eval_summary,
            policy_config=policy_config,
            env_processor_config=request.env_processor_config,
            action_processor_config=request.action_processor_config,
            training_lineage=request.training_lineage,
            metadata={
                **request.metadata,
                "policy_path": "policy",
                "runtime": request.metadata.get("runtime", {}),
            },
        )
        manifest.write_json(artifact_root / "manifest.json")
        self._write_json(artifact_root / "build_request.json", request.to_dict())
        return artifact_root

    def _write_stats(self, source_stats_path: str | Path | None, stats_dest: Path) -> str:
        if source_stats_path is None:
            self._write_json(stats_dest, {})
            return compute_sha256(stats_dest)

        source_stats_path = Path(source_stats_path)
        if not source_stats_path.exists():
            raise FileNotFoundError(f"Stats path does not exist: {source_stats_path}")
        shutil.copy2(source_stats_path, stats_dest)
        return compute_sha256(stats_dest)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


class FilesystemOpenPIArtifactBuilder:
    """Builds edge-compatible OpenPI deployment artifacts."""

    def build(self, request: OpenPIArtifactBuildRequest) -> Path:
        output_root = Path(request.output_root)
        artifact_root = output_root / request.artifact_id
        if artifact_root.exists():
            shutil.rmtree(artifact_root)
        artifact_root.mkdir(parents=True, exist_ok=True)

        policy_dir = artifact_root / "policy"
        policy_dir.mkdir()
        self._write_json(policy_dir / "config.json", {"type": "openpi"})

        stats_path = artifact_root / "stats.json"
        self._write_json(stats_path, {})

        env_processor_path = artifact_root / "env_processor_config.json"
        action_processor_path = artifact_root / "action_processor_config.json"
        self._write_json(env_processor_path, request.env_processor_config)
        self._write_json(action_processor_path, request.action_processor_config)

        runtime_metadata = {
            "backend": request.runtime_backend,
            "policy_ref": request.policy_ref,
            "action_horizon": request.action_horizon,
            "action_dim": request.action_dim,
            "model_output_key": request.model_output_key,
            "output_transform": request.output_transform,
        }
        runtime_metadata.update(request.runtime_metadata)
        if request.runtime_assets_dir is not None:
            source_assets_dir = Path(request.runtime_assets_dir)
            if not source_assets_dir.exists():
                raise FileNotFoundError(f"Runtime assets directory does not exist: {source_assets_dir}")
            copied_assets_dir = artifact_root / "runtime_assets"
            shutil.copytree(source_assets_dir, copied_assets_dir)
            runtime_metadata["runtime_assets_path"] = "runtime_assets"

        manifest = ArtifactManifest(
            artifact_id=request.artifact_id,
            base_checkpoint=request.base_checkpoint,
            runtime_abi_version=request.runtime_abi_version,
            observation_schema_version=request.observation_schema_version,
            action_schema_version=request.action_schema_version,
            env_processor_digest=compute_sha256(env_processor_path),
            action_processor_digest=compute_sha256(action_processor_path),
            stats_digest=compute_sha256(stats_path),
            policy_digest=compute_tree_digest(policy_dir),
            compatible_robot_types=request.compatible_robot_types,
            compatible_camera_layouts=request.compatible_camera_layouts,
            eval_summary=request.eval_summary,
            policy_config={"type": "openpi"},
            env_processor_config=request.env_processor_config,
            action_processor_config=request.action_processor_config,
            training_lineage=request.training_lineage,
            metadata={
                **request.metadata,
                "policy_path": "policy",
                "runtime": runtime_metadata,
                "observation_contract": request.observation_contract,
            },
        )
        manifest.write_json(artifact_root / "manifest.json")
        self._write_json(artifact_root / "build_request.json", request.to_dict())
        return artifact_root

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
