#!/usr/bin/env python

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from lerobot.cloud.materializer import MaterializedDatasetManifest, MaterializedEpisodeSummary

from .controller import ReleaseController
from .registry import ReleaseRegistry


@dataclass(frozen=True)
class ControllerRunResult:
    action: str
    dataset_fingerprint: str | None
    artifact_id: str | None
    channel: str
    state_path: str
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControllerRunState:
    dataset_fingerprint: str
    artifact_id: str
    channel: str
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AutoReleaseController:
    """Runs a single control-plane decision step from a materialized dataset snapshot."""

    def __init__(
        self,
        *,
        registry_root: str | Path,
        artifact_output_root: str | Path,
        state_root: str | Path,
    ):
        self.registry = ReleaseRegistry(registry_root)
        self.release_controller = ReleaseController(
            registry_root=registry_root,
            artifact_root=artifact_output_root,
        )
        self.artifact_output_root = Path(artifact_output_root)
        self.state_root = Path(state_root)
        self.state_root.mkdir(parents=True, exist_ok=True)

    def run_once(
        self,
        *,
        materialized_root: str | Path,
        train_output_root: str | Path,
        incident_root: str | Path,
        report_root: str | Path,
        channel: str,
        artifact_prefix: str,
        compatible_robot_types: list[str],
        compatible_camera_layouts: list[str],
        rollout_reason: str | None = None,
        report_filename: str = "rollout_status.json",
        device_state_roots: dict[str, str] | None = None,
        policy_type: str = "act",
        batch_size: int = 1,
    ) -> ControllerRunResult:
        manifest_path = Path(materialized_root) / "manifest.json"
        manifest = self._read_manifest(manifest_path)
        dataset_fingerprint = self._compute_manifest_fingerprint(manifest)
        state_path = self.state_root / f"{channel}.json"

        if manifest.episode_count == 0 or manifest.lerobot_root is None or manifest.lerobot_repo_id is None:
            report_path = self.release_controller.write_rollout_report(
                incident_root=incident_root,
                report_root=report_root,
                report_filename=report_filename,
                device_state_roots=device_state_roots,
            )
            return ControllerRunResult(
                action="no_data",
                dataset_fingerprint=dataset_fingerprint,
                artifact_id=None,
                channel=channel,
                state_path=str(state_path),
                report_path=str(report_path),
            )

        artifact_id = f"{artifact_prefix}-{dataset_fingerprint[:12]}"
        existing_state = self._read_state(state_path)
        channel_target = self.registry.get_channel_target(channel)

        if (
            existing_state is not None
            and existing_state.dataset_fingerprint == dataset_fingerprint
            and existing_state.artifact_id == artifact_id
            and channel_target is not None
            and channel_target.target_artifact_id == artifact_id
        ):
            report_path = self.release_controller.write_rollout_report(
                incident_root=incident_root,
                report_root=report_root,
                report_filename=report_filename,
                device_state_roots=device_state_roots,
            )
            state = ControllerRunState(
                dataset_fingerprint=dataset_fingerprint,
                artifact_id=artifact_id,
                channel=channel,
                report_path=str(report_path),
            )
            self._write_state(state_path, state)
            return ControllerRunResult(
                action="noop",
                dataset_fingerprint=dataset_fingerprint,
                artifact_id=artifact_id,
                channel=channel,
                state_path=str(state_path),
                report_path=str(report_path),
            )

        from lerobot.scripts.control_plane_release_cycle import run_release_cycle

        result = run_release_cycle(
            dataset_root=manifest.lerobot_root,
            repo_id=manifest.lerobot_repo_id,
            train_output_dir=str(Path(train_output_root) / artifact_id),
            artifact_output_root=str(self.artifact_output_root),
            registry_root=str(self.registry.layout.root),
            artifact_id=artifact_id,
            base_checkpoint=artifact_id,
            channel=channel,
            incident_root=str(incident_root),
            report_root=str(report_root),
            report_filename=report_filename,
            compatible_robot_types=compatible_robot_types,
            compatible_camera_layouts=compatible_camera_layouts,
            rollout_reason=rollout_reason,
            device_state_roots=device_state_roots,
            policy_type=policy_type,
            batch_size=batch_size,
        )
        state = ControllerRunState(
            dataset_fingerprint=dataset_fingerprint,
            artifact_id=artifact_id,
            channel=channel,
            report_path=str(result["report_path"]),
        )
        self._write_state(state_path, state)
        return ControllerRunResult(
            action="released",
            dataset_fingerprint=dataset_fingerprint,
            artifact_id=artifact_id,
            channel=channel,
            state_path=str(state_path),
            report_path=str(result["report_path"]),
        )

    def _compute_manifest_fingerprint(self, manifest: MaterializedDatasetManifest) -> str:
        payload = {
            "episode_count": manifest.episode_count,
            "total_step_count": manifest.total_step_count,
            "repo_id": manifest.lerobot_repo_id,
            "episodes": [
                {
                    "episode_id": episode.episode_id,
                    "step_count": episode.step_count,
                    "policy_artifact_id": episode.policy_artifact_id,
                    "task_id": episode.task_id,
                    "robot_id": episode.robot_id,
                    "source_commit_path": episode.source_commit_path,
                }
                for episode in manifest.episodes
            ],
        }
        return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _read_manifest(self, path: Path) -> MaterializedDatasetManifest:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return MaterializedDatasetManifest(
            episode_count=payload["episode_count"],
            total_step_count=payload["total_step_count"],
            episodes=[MaterializedEpisodeSummary(**episode) for episode in payload["episodes"]],
            lerobot_repo_id=payload.get("lerobot_repo_id"),
            lerobot_root=payload.get("lerobot_root"),
        )

    def _read_state(self, path: Path) -> ControllerRunState | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return ControllerRunState(**payload)

    def _write_state(self, path: Path, state: ControllerRunState) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
