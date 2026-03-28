#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.cloud.artifact_builder import FilesystemOpenPIArtifactBuilder, OpenPIArtifactBuildRequest
from lerobot.edge.model_manager import EdgeCompatibilityContext, EdgeModelManager
from lerobot.edge.openpi_loader import resolve_openpi_policy_ref
from lerobot.edge.openpi_runtime import OpenPIInProcessRuntime, OpenPIInProcessRuntimeConfig, OpenPIObservationAdapterConfig
from lerobot.edge.runner import EdgeRobotRunner
from lerobot.edge.spool import EdgeEpisodeSpool


@dataclass
class OpenPIEdgeSmokeRobot:
    observation: dict[str, Any]
    action_dim: int
    id: str = "openpi-smoke-robot"
    name: str = "openpi_smoke_robot"

    @property
    def action_features(self) -> dict[str, type]:
        return {f"motor_{idx + 1}.pos": float for idx in range(self.action_dim)}

    def connect(self) -> None:
        return None

    def get_observation(self) -> dict[str, Any]:
        copied: dict[str, Any] = {}
        for key, value in self.observation.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            else:
                copied[key] = value
        return copied

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        return action

    def disconnect(self) -> None:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real OpenPI in-process edge runner smoke.")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to the OpenPI checkpoint directory.")
    parser.add_argument("--observation-npz", required=True, help="Path to the sample OpenPI observation npz.")
    parser.add_argument("--observation-meta-json", required=True, help="Path to the sample observation metadata json.")
    parser.add_argument("--spool-root", required=True, help="Where to write the edge spool.")
    parser.add_argument("--artifact-root", help="Where to build and load the temporary OpenPI artifact.")
    parser.add_argument("--policy-config", default="pi0_aloha_sim", help="OpenPI config name.")
    parser.add_argument("--task-id", default="aloha_transfer_cube", help="Task id recorded in the episode.")
    parser.add_argument("--channel", default="staging", help="Release channel recorded in the episode.")
    parser.add_argument("--robot-type", default="mock_robot", help="Compatibility robot_type for the artifact.")
    parser.add_argument(
        "--camera-layout",
        default="single_arm_mock",
        help="Compatibility camera_layout for the artifact.",
    )
    parser.add_argument("--max-steps", type=int, default=2, help="How many runner steps to execute.")
    parser.add_argument("--fps", type=int, default=2, help="Runner loop frequency.")
    return parser.parse_args()


def _load_observation(npz_path: Path, meta_json_path: Path) -> dict[str, Any]:
    sample = np.load(npz_path)
    with meta_json_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    observation = {key: sample[key] for key in sample.files}
    observation["prompt"] = meta["prompt"]
    return observation


def run_openpi_edge_smoke(
    *,
    checkpoint_dir: Path,
    observation_npz: Path,
    observation_meta_json: Path,
    spool_root: Path,
    artifact_root: Path | None = None,
    policy_config: str = "pi0_aloha_sim",
    task_id: str = "aloha_transfer_cube",
    channel: str = "staging",
    robot_type: str = "mock_robot",
    camera_layout: str = "single_arm_mock",
    max_steps: int = 2,
    fps: int = 2,
) -> dict[str, Any]:
    observation = _load_observation(observation_npz, observation_meta_json)
    policy_ref = f"openpi://checkpoint?config={policy_config}&dir={checkpoint_dir}"
    observation_contract = {
        "mode": "aloha_raw",
        "state_key": "state",
        "state_dim": 14,
        "image_keys": [
            "image.base_0_rgb",
            "image.left_wrist_0_rgb",
            "image.right_wrist_0_rgb",
        ],
        "image_aliases": {
            "image.base_0_rgb": "cam_high",
            "image.left_wrist_0_rgb": "cam_left_wrist",
            "image.right_wrist_0_rgb": "cam_right_wrist",
        },
        "prompt_key": "prompt",
        "transpose_images_to_chw": True,
    }

    artifact_id = f"openpi-smoke-{policy_config}"
    if artifact_root is None:
        policy = resolve_openpi_policy_ref(policy_ref)
        runtime = OpenPIInProcessRuntime(
            OpenPIInProcessRuntimeConfig(
                action_horizon=50,
                action_dim=14,
                policy_ref=policy_ref,
                observation=OpenPIObservationAdapterConfig(**observation_contract),
            ),
            policy=policy,
        )
    else:
        builder = FilesystemOpenPIArtifactBuilder()
        builder.build(
            OpenPIArtifactBuildRequest(
                artifact_id=artifact_id,
                base_checkpoint=policy_config,
                output_root=artifact_root,
                compatible_robot_types=[robot_type],
                compatible_camera_layouts=[camera_layout],
                policy_ref=policy_ref,
                runtime_backend="in_process",
                action_horizon=50,
                action_dim=14,
                observation_contract=observation_contract,
                metadata={"task": task_id, "robot_type": robot_type},
            )
        )
        manager = EdgeModelManager(artifact_root)
        artifact = manager.load_artifact_by_id(artifact_id)
        errors = manager.validate_artifact(
            artifact,
            EdgeCompatibilityContext(
                robot_type=robot_type,
                camera_layout=camera_layout,
            ),
        )
        if errors:
            raise ValueError("; ".join(errors))
        runtime = manager.activate_artifact(
            artifact,
            warmup_observation=observation,
        )

    robot = OpenPIEdgeSmokeRobot(observation=observation, action_dim=14)
    spool = EdgeEpisodeSpool(spool_root)
    runner = EdgeRobotRunner(
        robot=robot,
        runtime=runtime,
        spool=spool,
        policy_artifact_id=artifact_id,
        processor_bundle_id="openpi-aloha-raw-v1",
        task_id=task_id,
        channel=channel,
        action_processor=None,
    )

    robot.connect()
    try:
        result = runner.run_episode(max_steps=max_steps, fps=fps)
    finally:
        robot.disconnect()

    sealed_dir = spool_root / "sealed" / result.episode_id
    with (sealed_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with (sealed_dir / "steps.jsonl").open("r", encoding="utf-8") as handle:
        first_step = json.loads(handle.readline())

    action_names = list(robot.action_features)
    return {
        "episode_id": result.episode_id,
        "step_count": result.step_count,
        "duration_s": result.duration_s,
        "stop_reason": result.stop_reason,
        "summary": summary,
        "first_action": [first_step["action"][name] for name in action_names],
        "spool_dir": str(sealed_dir),
        "artifact_id": artifact_id,
        "artifact_root": None if artifact_root is None else str(artifact_root / artifact_id),
    }


def main() -> None:
    args = _parse_args()
    result = run_openpi_edge_smoke(
        checkpoint_dir=Path(args.checkpoint_dir),
        observation_npz=Path(args.observation_npz),
        observation_meta_json=Path(args.observation_meta_json),
        spool_root=Path(args.spool_root),
        artifact_root=None if args.artifact_root is None else Path(args.artifact_root),
        policy_config=args.policy_config,
        task_id=args.task_id,
        channel=args.channel,
        robot_type=args.robot_type,
        camera_layout=args.camera_layout,
        max_steps=args.max_steps,
        fps=args.fps,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
