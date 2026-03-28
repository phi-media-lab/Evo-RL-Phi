#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.edge.openpi_loader import resolve_openpi_hybrid_bridge_feed_builder_with_progress
from lerobot.edge.openpi_runtime import OpenPIHybridBridgeRuntimeConfig, OpenPIObservationAdapter, OpenPIObservationAdapterConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a real OpenPI hybrid bridge feed and report staged timings.")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to the converted PyTorch OpenPI checkpoint.")
    parser.add_argument("--observation-npz", required=True, help="Path to the sample OpenPI observation npz.")
    parser.add_argument("--observation-meta-json", required=True, help="Path to the sample observation metadata json.")
    parser.add_argument("--policy-config", default="pi0_aloha_sim", help="OpenPI config name.")
    parser.add_argument("--bridge-mode", default="mlx_full_prefix_hostbridge", help="Bridge backend mode.")
    parser.add_argument("--prefix-mode", default="observation", help="Prefix preparation mode.")
    parser.add_argument("--export-precision", default="float16", help="Export precision for the bridge feed.")
    parser.add_argument("--device", default="cpu", help="PyTorch device for feed preparation.")
    parser.add_argument("--openpi-repo-root", default="/Users/fbsh/ane-openpi/openpi", help="ane-openpi repository root.")
    parser.add_argument("--output-json", help="Optional path to write the timing summary json.")
    return parser.parse_args()


def _write_progress(output_json: Path | None, payload: dict[str, Any]) -> None:
    if output_json is None:
        return
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_observation(npz_path: Path, meta_json_path: Path) -> dict[str, Any]:
    sample = np.load(npz_path)
    with meta_json_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    observation = {key: sample[key] for key in sample.files}
    observation["prompt"] = meta["prompt"]
    return observation


def run_openpi_bridge_feed_smoke(
    *,
    checkpoint_dir: Path,
    observation_npz: Path,
    observation_meta_json: Path,
    policy_config: str,
    bridge_mode: str,
    prefix_mode: str,
    export_precision: str,
    device: str,
    openpi_repo_root: Path,
    output_json: Path | None = None,
) -> dict[str, Any]:
    timings: dict[str, float] = {}
    result: dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "policy_config": policy_config,
        "bridge_mode": bridge_mode,
        "prefix_mode": prefix_mode,
        "export_precision": export_precision,
        "device": device,
        "stage": "starting",
        "timings": timings,
    }
    _write_progress(output_json, result)

    started = time.perf_counter()
    observation = _load_observation(observation_npz, observation_meta_json)
    timings["load_observation_s"] = round(time.perf_counter() - started, 6)
    result["stage"] = "observation_loaded"
    _write_progress(output_json, result)

    observation_contract = OpenPIObservationAdapterConfig(
        mode="openpi_raw",
        state_key="state",
        image_keys=[
            "image.base_0_rgb",
            "image.left_wrist_0_rgb",
            "image.right_wrist_0_rgb",
        ],
        prompt_key="prompt",
    )
    adapter = OpenPIObservationAdapter(observation_contract)

    started = time.perf_counter()
    adapted = adapter.adapt(observation)
    timings["adapt_observation_s"] = round(time.perf_counter() - started, 6)
    result["stage"] = "observation_adapted"
    _write_progress(output_json, result)

    runtime_config = OpenPIHybridBridgeRuntimeConfig(
        model_path="/tmp/fake.mlpackage",
        bridge_mode=bridge_mode,
        action_horizon=50,
        action_dim=14,
        observation=observation_contract,
        prefix_mode=prefix_mode,
        config_name=policy_config,
        checkpoint_dir=str(checkpoint_dir),
        export_precision=export_precision,
        device=device,
        openpi_repo_root=str(openpi_repo_root),
    )

    started = time.perf_counter()
    def on_progress(stage: str, stage_timings: dict[str, float]) -> None:
        result["stage"] = stage
        result["timings"] = dict(stage_timings)
        _write_progress(output_json, result)

    feed_builder = resolve_openpi_hybrid_bridge_feed_builder_with_progress(
        openpi_repo_root=str(openpi_repo_root),
        config_name=policy_config,
        checkpoint_dir=str(checkpoint_dir),
        device=device,
        prefix_mode=prefix_mode,
        export_precision=export_precision,
        on_progress=on_progress,
    )
    timings["resolve_feed_builder_s"] = round(time.perf_counter() - started, 6)
    result["stage"] = "feed_builder_resolved"
    _write_progress(output_json, result)

    started = time.perf_counter()
    feed = feed_builder(adapted, runtime_config)
    timings["build_feed_s"] = round(time.perf_counter() - started, 6)
    result["stage"] = "completed"
    result["feed_shapes"] = {key: list(value.shape) for key, value in feed.items()}
    result["feed_dtypes"] = {key: str(value.dtype) for key, value in feed.items()}
    _write_progress(output_json, result)
    return result


def main() -> None:
    args = _parse_args()
    result = run_openpi_bridge_feed_smoke(
        checkpoint_dir=Path(args.checkpoint_dir),
        observation_npz=Path(args.observation_npz),
        observation_meta_json=Path(args.observation_meta_json),
        policy_config=args.policy_config,
        bridge_mode=args.bridge_mode,
        prefix_mode=args.prefix_mode,
        export_precision=args.export_precision,
        device=args.device,
        openpi_repo_root=Path(args.openpi_repo_root),
        output_json=None if args.output_json is None else Path(args.output_json),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
