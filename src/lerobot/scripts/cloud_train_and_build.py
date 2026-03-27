#!/usr/bin/env python

from __future__ import annotations

import argparse

from lerobot.cloud.artifact_builder import ArtifactBuildRequest, FilesystemArtifactBuilder
from lerobot.scripts.cloud_train_smoke import run_training_smoke


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one-step training smoke and build an edge deployment artifact."
    )
    parser.add_argument("--dataset-root", required=True, help="Local LeRobotDataset root.")
    parser.add_argument("--repo-id", required=True, help="Repo id metadata of the local LeRobotDataset.")
    parser.add_argument("--train-output-dir", required=True, help="Scratch output dir for the smoke train.")
    parser.add_argument("--artifact-output-root", required=True, help="Artifact output root.")
    parser.add_argument("--artifact-id", required=True, help="Artifact id / output directory name.")
    parser.add_argument("--base-checkpoint", required=True, help="Checkpoint or lineage id for the artifact.")
    parser.add_argument("--robot-type", action="append", required=True, dest="robot_types")
    parser.add_argument("--camera-layout", action="append", required=True, dest="camera_layouts")
    parser.add_argument("--policy-type", default="act")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser


def run_train_and_build(
    *,
    dataset_root: str,
    repo_id: str,
    train_output_dir: str,
    artifact_output_root: str,
    artifact_id: str,
    base_checkpoint: str,
    compatible_robot_types: list[str],
    compatible_camera_layouts: list[str],
    policy_type: str = "act",
    batch_size: int = 1,
) -> dict[str, float | int | str]:
    train_result = run_training_smoke(
        dataset_root=dataset_root,
        repo_id=repo_id,
        output_dir=train_output_dir,
        policy_type=policy_type,
        batch_size=batch_size,
        save_policy=True,
    )
    artifact_root = FilesystemArtifactBuilder().build(
        ArtifactBuildRequest(
            artifact_id=artifact_id,
            base_checkpoint=base_checkpoint,
            policy_dir=str(train_result["policy_dir"]),
            output_root=artifact_output_root,
            compatible_robot_types=compatible_robot_types,
            compatible_camera_layouts=compatible_camera_layouts,
            stats_path=str(train_result["stats_path"]),
            eval_summary={"loss": float(train_result["loss"])},
            training_lineage={
                "repo_id": repo_id,
                "policy_type": policy_type,
                "train_output_dir": train_output_dir,
            },
        )
    )
    return {
        **train_result,
        "artifact_id": artifact_id,
        "artifact_root": str(artifact_root),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_train_and_build(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        train_output_dir=args.train_output_dir,
        artifact_output_root=args.artifact_output_root,
        artifact_id=args.artifact_id,
        base_checkpoint=args.base_checkpoint,
        compatible_robot_types=args.robot_types,
        compatible_camera_layouts=args.camera_layouts,
        policy_type=args.policy_type,
        batch_size=args.batch_size,
    )
    print(
        f"train-and-build artifact={result['artifact_id']} policy={result['policy_type']} "
        f"episodes={result['dataset_num_episodes']} frames={result['dataset_num_frames']} "
        f"loss={result['loss']:.6f}"
    )


if __name__ == "__main__":
    main()
