#!/usr/bin/env python

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import logging
from pathlib import Path
import time
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
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    duration_s: float | None = None

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


@dataclass(frozen=True)
class ControllerLoopResult:
    iterations: int
    results: list[ControllerRunResult]
    history_path: str
    latest_path: str
    metrics_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "results": [result.to_dict() for result in self.results],
            "history_path": self.history_path,
            "latest_path": self.latest_path,
            "metrics_path": self.metrics_path,
        }


@dataclass(frozen=True)
class ControllerLoopMetrics:
    iterations: int
    waiting_count: int = 0
    no_data_count: int = 0
    noop_count: int = 0
    released_count: int = 0
    last_action: str | None = None
    last_artifact_id: str | None = None
    last_completed_at_utc: str | None = None

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


class AutoReleaseDaemon:
    """Polling wrapper around AutoReleaseController for long-running control-plane automation."""

    def __init__(
        self,
        *,
        registry_root: str | Path,
        artifact_output_root: str | Path,
        state_root: str | Path,
        runtime_root: str | Path,
    ):
        self.controller = AutoReleaseController(
            registry_root=registry_root,
            artifact_output_root=artifact_output_root,
            state_root=state_root,
        )
        self.runtime_root = Path(runtime_root)
        self.runtime_root.mkdir(parents=True, exist_ok=True)

    def run(
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
        poll_interval_s: float = 5.0,
        max_iterations: int | None = None,
        sleep_fn: Any = time.sleep,
    ) -> ControllerLoopResult:
        results: list[ControllerRunResult] = []
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            result = self._run_iteration(
                materialized_root=materialized_root,
                train_output_root=train_output_root,
                incident_root=incident_root,
                report_root=report_root,
                channel=channel,
                artifact_prefix=artifact_prefix,
                compatible_robot_types=compatible_robot_types,
                compatible_camera_layouts=compatible_camera_layouts,
                rollout_reason=rollout_reason,
                report_filename=report_filename,
                device_state_roots=device_state_roots,
                policy_type=policy_type,
                batch_size=batch_size,
            )
            results.append(result)
            self._append_history(result)
            self._write_latest(result)
            self._write_metrics(results)
            logging.info(
                "auto-release iteration=%s action=%s artifact=%s duration_s=%s",
                iteration,
                result.action,
                result.artifact_id,
                result.duration_s,
            )
            if max_iterations is not None and iteration >= max_iterations:
                break
            sleep_fn(poll_interval_s)

        return ControllerLoopResult(
            iterations=iteration,
            results=results,
            history_path=str(self.runtime_root / "history.jsonl"),
            latest_path=str(self.runtime_root / "latest.json"),
            metrics_path=str(self.runtime_root / "metrics.json"),
        )

    def _run_iteration(
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
        rollout_reason: str | None,
        report_filename: str,
        device_state_roots: dict[str, str] | None,
        policy_type: str,
        batch_size: int,
    ) -> ControllerRunResult:
        started_at = self._utc_now()
        started_perf = time.perf_counter()
        manifest_path = Path(materialized_root) / "manifest.json"
        if not manifest_path.exists():
            return ControllerRunResult(
                action="waiting",
                dataset_fingerprint=None,
                artifact_id=None,
                channel=channel,
                state_path=str(self.controller.state_root / f"{channel}.json"),
                report_path=None,
                started_at_utc=started_at,
                completed_at_utc=self._utc_now(),
                duration_s=round(time.perf_counter() - started_perf, 6),
            )
        result = self.controller.run_once(
            materialized_root=materialized_root,
            train_output_root=train_output_root,
            incident_root=incident_root,
            report_root=report_root,
            channel=channel,
            artifact_prefix=artifact_prefix,
            compatible_robot_types=compatible_robot_types,
            compatible_camera_layouts=compatible_camera_layouts,
            rollout_reason=rollout_reason,
            report_filename=report_filename,
            device_state_roots=device_state_roots,
            policy_type=policy_type,
            batch_size=batch_size,
        )
        return ControllerRunResult(
            action=result.action,
            dataset_fingerprint=result.dataset_fingerprint,
            artifact_id=result.artifact_id,
            channel=result.channel,
            state_path=result.state_path,
            report_path=result.report_path,
            started_at_utc=started_at,
            completed_at_utc=self._utc_now(),
            duration_s=round(time.perf_counter() - started_perf, 6),
        )

    def _append_history(self, result: ControllerRunResult) -> None:
        history_path = self.runtime_root / "history.jsonl"
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), sort_keys=True))
            handle.write("\n")

    def _write_latest(self, result: ControllerRunResult) -> None:
        latest_path = self.runtime_root / "latest.json"
        with latest_path.open("w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _write_metrics(self, results: list[ControllerRunResult]) -> None:
        metrics = ControllerLoopMetrics(
            iterations=len(results),
            waiting_count=sum(1 for result in results if result.action == "waiting"),
            no_data_count=sum(1 for result in results if result.action == "no_data"),
            noop_count=sum(1 for result in results if result.action == "noop"),
            released_count=sum(1 for result in results if result.action == "released"),
            last_action=None if not results else results[-1].action,
            last_artifact_id=None if not results else results[-1].artifact_id,
            last_completed_at_utc=None if not results else results[-1].completed_at_utc,
        )
        metrics_path = self.runtime_root / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
