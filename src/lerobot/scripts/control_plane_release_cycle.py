#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.control_plane.controller import ReleaseController
from lerobot.scripts.cloud_train_build_and_release import run_train_build_and_release


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run train/build/release and emit a rollout report snapshot."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--train-output-dir", required=True)
    parser.add_argument("--artifact-output-root", required=True)
    parser.add_argument("--registry-root", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--incident-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--report-filename", default="rollout_status.json")
    parser.add_argument("--rollout-reason")
    parser.add_argument("--robot-type", action="append", required=True, dest="robot_types")
    parser.add_argument("--camera-layout", action="append", required=True, dest="camera_layouts")
    parser.add_argument("--device-state-root", action="append", default=[], dest="device_state_roots")
    parser.add_argument("--policy-type", default="act")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser


def _parse_device_state_roots(values: list[str]) -> dict[str, str]:
    roots: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"device-state-root must look like <device_id>=<path>, got: {value}")
        device_id, root = value.split("=", 1)
        roots[device_id] = root
    return roots


def run_release_cycle(
    *,
    dataset_root: str,
    repo_id: str,
    train_output_dir: str,
    artifact_output_root: str,
    registry_root: str,
    artifact_id: str,
    base_checkpoint: str,
    channel: str,
    incident_root: str,
    report_root: str,
    compatible_robot_types: list[str],
    compatible_camera_layouts: list[str],
    report_filename: str = "rollout_status.json",
    rollout_reason: str | None = None,
    device_state_roots: dict[str, str] | None = None,
    policy_type: str = "act",
    batch_size: int = 1,
) -> dict[str, float | int | str | None]:
    release_result = run_train_build_and_release(
        dataset_root=dataset_root,
        repo_id=repo_id,
        train_output_dir=train_output_dir,
        artifact_output_root=artifact_output_root,
        registry_root=registry_root,
        artifact_id=artifact_id,
        base_checkpoint=base_checkpoint,
        channel=channel,
        compatible_robot_types=compatible_robot_types,
        compatible_camera_layouts=compatible_camera_layouts,
        rollout_reason=rollout_reason,
        policy_type=policy_type,
        batch_size=batch_size,
    )
    report_path = ReleaseController(
        registry_root=registry_root,
        artifact_root=artifact_output_root,
    ).write_rollout_report(
        incident_root=incident_root,
        report_root=report_root,
        report_filename=report_filename,
        device_state_roots=None if device_state_roots is None else {k: Path(v) for k, v in device_state_roots.items()},
    )
    return {
        **release_result,
        "report_path": str(report_path),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_release_cycle(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        train_output_dir=args.train_output_dir,
        artifact_output_root=args.artifact_output_root,
        registry_root=args.registry_root,
        artifact_id=args.artifact_id,
        base_checkpoint=args.base_checkpoint,
        channel=args.channel,
        incident_root=args.incident_root,
        report_root=args.report_root,
        report_filename=args.report_filename,
        rollout_reason=args.rollout_reason,
        compatible_robot_types=args.robot_types,
        compatible_camera_layouts=args.camera_layouts,
        device_state_roots=_parse_device_state_roots(args.device_state_roots),
        policy_type=args.policy_type,
        batch_size=args.batch_size,
    )
    print(
        f"release-cycle artifact={result['artifact_id']} channel={result['channel']} "
        f"report={result['report_path']} loss={result['loss']:.6f}"
    )


if __name__ == "__main__":
    main()
