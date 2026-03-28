#!/usr/bin/env python

from __future__ import annotations

import argparse
import json

from lerobot.cloud.artifact_builder import FilesystemOpenPIArtifactBuilder, OpenPIArtifactBuildRequest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an edge deployment artifact for an OpenPI backend.")
    parser.add_argument("--artifact-id", required=True, help="Artifact id / output directory name.")
    parser.add_argument("--base-checkpoint", required=True, help="Training checkpoint or source lineage id.")
    parser.add_argument("--output-root", required=True, help="Artifact output root.")
    parser.add_argument("--policy-ref", required=True, help="OpenPI policy_ref, e.g. openpi://checkpoint?... .")
    parser.add_argument(
        "--runtime-backend",
        required=True,
        choices=["in_process", "websocket", "coreml", "hybrid_bridge_v1"],
    )
    parser.add_argument("--action-horizon", required=True, type=int)
    parser.add_argument("--action-dim", required=True, type=int)
    parser.add_argument("--model-output-key", default="", help="Optional predictor output key for rollout/coreml models.")
    parser.add_argument(
        "--output-transform",
        default="",
        help="Optional output transform, e.g. 'aloha_actions' for rollout models.",
    )
    parser.add_argument("--robot-type", action="append", required=True, dest="robot_types")
    parser.add_argument("--camera-layout", action="append", required=True, dest="camera_layouts")
    parser.add_argument(
        "--observation-contract-json",
        required=True,
        help="JSON file describing the OpenPI observation contract.",
    )
    parser.add_argument("--runtime-assets-dir", help="Optional runtime assets directory to bundle into the artifact.")
    parser.add_argument(
        "--runtime-metadata-json",
        help="Optional JSON file with extra runtime metadata, e.g. bridge_mode/config_name/checkpoint_dir.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    with open(args.observation_contract_json, "r", encoding="utf-8") as handle:
        observation_contract = json.load(handle)
    runtime_metadata = {}
    if args.runtime_metadata_json is not None:
        with open(args.runtime_metadata_json, "r", encoding="utf-8") as handle:
            runtime_metadata = json.load(handle)

    artifact_root = FilesystemOpenPIArtifactBuilder().build(
        OpenPIArtifactBuildRequest(
            artifact_id=args.artifact_id,
            base_checkpoint=args.base_checkpoint,
            output_root=args.output_root,
            compatible_robot_types=args.robot_types,
            compatible_camera_layouts=args.camera_layouts,
            policy_ref=args.policy_ref,
            runtime_backend=args.runtime_backend,
            action_horizon=args.action_horizon,
            action_dim=args.action_dim,
            model_output_key=args.model_output_key,
            output_transform=args.output_transform,
            runtime_metadata=runtime_metadata,
            observation_contract=observation_contract,
            runtime_assets_dir=args.runtime_assets_dir,
        )
    )
    print(str(artifact_root))


if __name__ == "__main__":
    main()
