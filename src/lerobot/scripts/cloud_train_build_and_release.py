#!/usr/bin/env python

from __future__ import annotations

import argparse

from lerobot.control_plane.controller import ReleaseController
from lerobot.scripts.cloud_train_and_build import run_train_and_build


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run smoke train, build an artifact, and publish it to a release channel."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--train-output-dir", required=True)
    parser.add_argument("--artifact-output-root", required=True)
    parser.add_argument("--registry-root", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--rollout-reason")
    parser.add_argument("--robot-type", action="append", required=True, dest="robot_types")
    parser.add_argument("--camera-layout", action="append", required=True, dest="camera_layouts")
    parser.add_argument("--policy-type", default="act")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser


def run_train_build_and_release(
    *,
    dataset_root: str,
    repo_id: str,
    train_output_dir: str,
    artifact_output_root: str,
    registry_root: str,
    artifact_id: str,
    base_checkpoint: str,
    channel: str,
    compatible_robot_types: list[str],
    compatible_camera_layouts: list[str],
    rollout_reason: str | None = None,
    policy_type: str = "act",
    batch_size: int = 1,
) -> dict[str, float | int | str | None]:
    build_result = run_train_and_build(
        dataset_root=dataset_root,
        repo_id=repo_id,
        train_output_dir=train_output_dir,
        artifact_output_root=artifact_output_root,
        artifact_id=artifact_id,
        base_checkpoint=base_checkpoint,
        compatible_robot_types=compatible_robot_types,
        compatible_camera_layouts=compatible_camera_layouts,
        policy_type=policy_type,
        batch_size=batch_size,
    )
    publish_result = ReleaseController(
        registry_root=registry_root,
        artifact_root=artifact_output_root,
    ).publish_artifact_to_channel(
        channel=channel,
        artifact_id=artifact_id,
        rollout_reason=rollout_reason,
    )
    return {
        **build_result,
        "channel": publish_result.channel,
        "registry_root": publish_result.registry_root,
        "rollout_reason": publish_result.rollout_reason,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_train_build_and_release(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        train_output_dir=args.train_output_dir,
        artifact_output_root=args.artifact_output_root,
        registry_root=args.registry_root,
        artifact_id=args.artifact_id,
        base_checkpoint=args.base_checkpoint,
        channel=args.channel,
        compatible_robot_types=args.robot_types,
        compatible_camera_layouts=args.camera_layouts,
        rollout_reason=args.rollout_reason,
        policy_type=args.policy_type,
        batch_size=args.batch_size,
    )
    print(
        f"train-build-release artifact={result['artifact_id']} channel={result['channel']} "
        f"episodes={result['dataset_num_episodes']} frames={result['dataset_num_frames']} "
        f"loss={result['loss']:.6f}"
    )


if __name__ == "__main__":
    main()
