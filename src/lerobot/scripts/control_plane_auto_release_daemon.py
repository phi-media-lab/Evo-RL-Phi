#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lerobot.control_plane.runner import AutoReleaseDaemon
from lerobot.utils.utils import init_logging


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a polling auto-release loop from a materialized dataset root."
    )
    parser.add_argument("--materialized-root", required=True)
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
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--max-iterations", type=int)
    return parser


def _parse_device_state_roots(values: list[str]) -> dict[str, str]:
    roots: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"device-state-root must look like <device_id>=<path>, got: {value}")
        device_id, root = value.split("=", 1)
        roots[device_id] = root
    return roots


def main() -> None:
    args = build_arg_parser().parse_args()
    runtime_root = Path(args.runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    init_logging(log_file=runtime_root / "daemon.log")
    logging.info("starting auto release daemon for channel=%s", args.channel)
    result = AutoReleaseDaemon(
        registry_root=args.registry_root,
        artifact_output_root=args.artifact_output_root,
        state_root=args.state_root,
        runtime_root=runtime_root,
    ).run(
        materialized_root=args.materialized_root,
        train_output_root=args.train_output_root,
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
        poll_interval_s=args.poll_interval_s,
        max_iterations=args.max_iterations,
    )
    latest = result.results[-1] if result.results else None
    logging.info(
        "auto release daemon finished iterations=%s latest_action=%s latest_artifact=%s metrics=%s",
        result.iterations,
        None if latest is None else latest.action,
        None if latest is None else latest.artifact_id,
        result.metrics_path,
    )
    print(
        f"auto-release-daemon iterations={result.iterations} "
        f"latest_action={None if latest is None else latest.action} "
        f"latest_artifact={None if latest is None else latest.artifact_id} "
        f"metrics={result.metrics_path}"
    )


if __name__ == "__main__":
    main()
