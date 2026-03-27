#!/usr/bin/env python

from __future__ import annotations

import argparse

from lerobot.cloud.artifact_builder import ArtifactBuildRequest, FilesystemArtifactBuilder


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an edge deployment artifact from a local policy dir.")
    parser.add_argument("--artifact-id", required=True, help="Artifact id / output directory name.")
    parser.add_argument("--base-checkpoint", required=True, help="Training checkpoint or source lineage id.")
    parser.add_argument("--policy-dir", required=True, help="Local pretrained policy directory.")
    parser.add_argument("--output-root", required=True, help="Artifact output root.")
    parser.add_argument("--robot-type", action="append", required=True, dest="robot_types")
    parser.add_argument("--camera-layout", action="append", required=True, dest="camera_layouts")
    parser.add_argument("--stats-path", help="Optional stats.json path to bundle into the artifact.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    builder = FilesystemArtifactBuilder()
    artifact_root = builder.build(
        ArtifactBuildRequest(
            artifact_id=args.artifact_id,
            base_checkpoint=args.base_checkpoint,
            policy_dir=args.policy_dir,
            output_root=args.output_root,
            compatible_robot_types=args.robot_types,
            compatible_camera_layouts=args.camera_layouts,
            stats_path=args.stats_path,
        )
    )
    print(str(artifact_root))


if __name__ == "__main__":
    main()
