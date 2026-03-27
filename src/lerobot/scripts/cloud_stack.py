#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from lerobot.cloud.ingestion import EpisodeIngestionHTTPServer, FilesystemEpisodeIngestionStore
from lerobot.cloud.materializer import MaterializerHTTPServer
from lerobot.scripts.control_plane_auto_release_daemon import run_auto_release_daemon


@dataclass(frozen=True)
class CloudServiceEndpoints:
    ingestion_base_url: str
    materializer_base_url: str


@dataclass
class CloudStackRuntimeStatus:
    phase: str = "starting"
    runtime_root: str | None = None
    endpoints: dict[str, str] = field(default_factory=dict)
    latest_action: str | None = None
    latest_artifact_id: str | None = None
    metrics_path: str | None = None
    latest_path: str | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CloudStackStatusHTTPRequestHandler(BaseHTTPRequestHandler):
    server: "CloudStackStatusHTTPServer"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            status = self.server.state.to_dict()
            code = HTTPStatus.OK if status["phase"] != "failed" else HTTPStatus.SERVICE_UNAVAILABLE
            self._write_json(code, {"ok": code == HTTPStatus.OK, "phase": status["phase"]})
            return
        if self.path == "/status":
            self._write_json(HTTPStatus.OK, self.server.state.to_dict())
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unsupported path: {self.path}"})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CloudStackStatusHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], state: CloudStackRuntimeStatus):
        self.state = state
        super().__init__(server_address, CloudStackStatusHTTPRequestHandler)


class CloudServiceGroup:
    def __init__(
        self,
        *,
        host: str,
        ingestion_port: int,
        materializer_port: int,
        ingestion_root: str,
        materialized_root: str,
        dataset_root: str,
    ):
        self.host = host
        self.ingestion_server = EpisodeIngestionHTTPServer(
            (host, ingestion_port),
            FilesystemEpisodeIngestionStore(ingestion_root),
        )
        self.materializer_server = MaterializerHTTPServer(
            (host, materializer_port),
            ingestion_root=ingestion_root,
            output_root=materialized_root,
            dataset_root=dataset_root,
        )
        self._ingestion_thread: threading.Thread | None = None
        self._materializer_thread: threading.Thread | None = None

    @property
    def endpoints(self) -> CloudServiceEndpoints:
        return CloudServiceEndpoints(
            ingestion_base_url=f"http://{self.host}:{self.ingestion_server.server_port}",
            materializer_base_url=f"http://{self.host}:{self.materializer_server.server_port}",
        )

    def start(self) -> CloudServiceEndpoints:
        self._ingestion_thread = threading.Thread(target=self.ingestion_server.serve_forever, daemon=True)
        self._materializer_thread = threading.Thread(target=self.materializer_server.serve_forever, daemon=True)
        self._ingestion_thread.start()
        self._materializer_thread.start()
        return self.endpoints

    def close(self) -> None:
        self.ingestion_server.shutdown()
        self.ingestion_server.server_close()
        self.materializer_server.shutdown()
        self.materializer_server.server_close()
        if self._ingestion_thread is not None:
            self._ingestion_thread.join(timeout=2)
        if self._materializer_thread is not None:
            self._materializer_thread.join(timeout=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal local cloud stack for Evo-RL.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ingestion-port", type=int, default=8000)
    parser.add_argument("--materializer-port", type=int, default=8001)
    parser.add_argument("--status-port", type=int, default=8002)
    parser.add_argument("--ingestion-root", required=True)
    parser.add_argument("--materialized-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-output-root", required=True)
    parser.add_argument("--artifact-output-root", required=True)
    parser.add_argument("--registry-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--incident-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--rollout-reason")
    parser.add_argument("--report-filename", default="rollout_status.json")
    parser.add_argument("--robot-type", action="append", required=True, dest="robot_types")
    parser.add_argument("--camera-layout", action="append", required=True, dest="camera_layouts")
    parser.add_argument("--device-state-root", action="append", default=[], dest="device_state_roots")
    parser.add_argument("--policy-type", default="act")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--materializer-repo-id", default="local/edge-materialized")
    parser.add_argument("--materializer-fps", type=int, default=20)
    parser.add_argument("--materializer-use-videos", action="store_true")
    parser.add_argument("--materializer-no-env-state-alias", action="store_true")
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--keep-alive", action="store_true")
    parser.add_argument("--idle-sleep-s", type=float, default=1.0)
    return parser


def _parse_device_state_roots(values: list[str]) -> dict[str, str]:
    roots: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"device-state-root must look like <device_id>=<path>, got: {value}")
        device_id, root = value.split("=", 1)
        roots[device_id] = root
    return roots


def run_cloud_stack(
    *,
    host: str,
    ingestion_port: int,
    materializer_port: int,
    status_port: int | None,
    ingestion_root: str,
    materialized_root: str,
    dataset_root: str,
    train_output_root: str,
    artifact_output_root: str,
    registry_root: str,
    state_root: str,
    runtime_root: str,
    incident_root: str,
    report_root: str,
    channel: str,
    artifact_prefix: str,
    compatible_robot_types: list[str],
    compatible_camera_layouts: list[str],
    rollout_reason: str | None = None,
    report_filename: str = "rollout_status.json",
    device_state_roots: dict[str, str] | None = None,
    policy_type: str = "act",
    batch_size: int = 1,
    materializer_repo_id: str = "local/edge-materialized",
    materializer_fps: int = 20,
    materializer_use_videos: bool = False,
    materializer_include_env_state_alias: bool = True,
    poll_interval_s: float = 5.0,
    max_iterations: int | None = None,
    keep_alive: bool = False,
    idle_sleep_s: float = 1.0,
    stop_event: threading.Event | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    runtime_root_path = Path(runtime_root)
    runtime_root_path.mkdir(parents=True, exist_ok=True)
    status = CloudStackRuntimeStatus(phase="starting", runtime_root=str(runtime_root_path))
    services = CloudServiceGroup(
        host=host,
        ingestion_port=ingestion_port,
        materializer_port=materializer_port,
        ingestion_root=ingestion_root,
        materialized_root=materialized_root,
        dataset_root=dataset_root,
    )
    status_server = None if status_port is None else CloudStackStatusHTTPServer((host, status_port), status)
    status_thread = None
    if status_server is not None:
        status_thread = threading.Thread(target=status_server.serve_forever, daemon=True)
        status_thread.start()
    endpoints = services.start()
    status.endpoints = {
        "ingestion_base_url": endpoints.ingestion_base_url,
        "materializer_base_url": endpoints.materializer_base_url,
    }
    if status_server is not None:
        status.endpoints["status_base_url"] = f"http://{host}:{status_server.server_port}"
    _write_status(runtime_root_path, status)
    status.phase = "running"
    _write_status(runtime_root_path, status)
    try:
        output = run_auto_release_daemon(
            materialized_root=materialized_root,
            train_output_root=train_output_root,
            artifact_output_root=artifact_output_root,
            registry_root=registry_root,
            state_root=state_root,
            runtime_root=runtime_root,
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
            materializer_base_url=endpoints.materializer_base_url,
            materializer_repo_id=materializer_repo_id,
            materializer_fps=materializer_fps,
            materializer_use_videos=materializer_use_videos,
            materializer_include_env_state_alias=materializer_include_env_state_alias,
            poll_interval_s=poll_interval_s,
            max_iterations=max_iterations,
        )
        latest_path = runtime_root_path / "latest.json"
        metrics_path = runtime_root_path / "metrics.json"
        if latest_path.exists():
            latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
            status.latest_action = latest_payload.get("action")
            status.latest_artifact_id = latest_payload.get("artifact_id")
            status.latest_path = str(latest_path)
        if metrics_path.exists():
            status.metrics_path = str(metrics_path)
        status.phase = "completed"
        _write_status(runtime_root_path, status)
        if keep_alive:
            status.phase = "serving"
            _write_status(runtime_root_path, status)
            guard = stop_event or threading.Event()
            while not guard.is_set():
                sleep_fn(idle_sleep_s)
        return output
    except Exception as exc:
        status.phase = "failed"
        status.last_error = str(exc)
        _write_status(runtime_root_path, status)
        raise
    finally:
        services.close()
        if status_server is not None:
            status_server.shutdown()
            status_server.server_close()
            if status_thread is not None:
                status_thread.join(timeout=2)


def _write_status(runtime_root: Path, status: CloudStackRuntimeStatus) -> None:
    with (runtime_root / "cloud_stack_status.json").open("w", encoding="utf-8") as handle:
        json.dump(status.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = build_arg_parser().parse_args()
    output = run_cloud_stack(
        host=args.host,
        ingestion_port=args.ingestion_port,
        materializer_port=args.materializer_port,
        status_port=args.status_port,
        ingestion_root=args.ingestion_root,
        materialized_root=args.materialized_root,
        dataset_root=args.dataset_root,
        train_output_root=args.train_output_root,
        artifact_output_root=args.artifact_output_root,
        registry_root=args.registry_root,
        state_root=args.state_root,
        runtime_root=args.runtime_root,
        incident_root=args.incident_root,
        report_root=args.report_root,
        channel=args.channel,
        artifact_prefix=args.artifact_prefix,
        compatible_robot_types=args.robot_types,
        compatible_camera_layouts=args.camera_layouts,
        rollout_reason=args.rollout_reason,
        report_filename=args.report_filename,
        device_state_roots=_parse_device_state_roots(args.device_state_roots),
        policy_type=args.policy_type,
        batch_size=args.batch_size,
        materializer_repo_id=args.materializer_repo_id,
        materializer_fps=args.materializer_fps,
        materializer_use_videos=args.materializer_use_videos,
        materializer_include_env_state_alias=not args.materializer_no_env_state_alias,
        poll_interval_s=args.poll_interval_s,
        max_iterations=args.max_iterations,
        keep_alive=args.keep_alive,
        idle_sleep_s=args.idle_sleep_s,
    )
    print(output)


if __name__ == "__main__":
    main()
