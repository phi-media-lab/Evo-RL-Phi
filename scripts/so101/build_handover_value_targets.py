#!/usr/bin/env python

"""Build lightweight SO101 handover value targets from LeRobot datasets and HIL manifests.

This script intentionally stays independent from the heavier Pistar06 value stack. It creates a
small parquet sidecar that can train a first-pass scalar value model from proprioceptive state.
The target is RECAP-inspired: successful states approach 0 as the episode nears completion, while
failed policy states are shifted down by a failure penalty.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SEGMENT_SUCCESS = {
    "success_policy": True,
    "human_correction": True,
    "recovery": True,
    "correction": True,
    "failed_policy_prefix": False,
    "failed_policy_no_correction": False,
    "failed_policy_suffix": False,
    "unknown": False,
}

DEFAULT_LOSS_WEIGHTS = {
    "success_policy": 1.0,
    "human_correction": 1.5,
    "recovery": 2.0,
    "correction": 2.0,
    "failed_policy_prefix": 0.75,
    "failed_policy_no_correction": 1.0,
    "failed_policy_suffix": 0.25,
    "unknown": 0.25,
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_root: Path
    manifest: Path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _parse_key_value_source(raw: str) -> SourceSpec:
    parts: dict[str, str] = {}
    for token in raw.split("::"):
        if "=" not in token:
            raise ValueError(
                f"Invalid --source '{raw}'. Expected 'name=...::root=...::manifest=...'."
            )
        key, value = token.split("=", maxsplit=1)
        parts[key.strip()] = value.strip()

    missing = {"root", "manifest"} - set(parts)
    if missing:
        raise ValueError(f"Invalid --source '{raw}'. Missing keys: {sorted(missing)}")

    root = Path(parts["root"]).expanduser()
    manifest = Path(parts["manifest"]).expanduser()
    name = parts.get("name") or root.name
    return SourceSpec(name=name, dataset_root=root, manifest=manifest)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def _load_dataset_frames(root: Path) -> pd.DataFrame:
    data_paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No parquet data files found under {root / 'data'}")
    frames = [pd.read_parquet(path) for path in data_paths]
    df = pd.concat(frames, ignore_index=True)
    required = {"episode_index", "frame_index", "index", "timestamp", "observation.state"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Dataset {root} is missing required columns: {sorted(missing)}")
    return df.sort_values(["episode_index", "frame_index", "index"]).reset_index(drop=True)


def _load_info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def _episode_success(row: dict[str, Any]) -> bool:
    if row.get("handover_success") is not None:
        return bool(row["handover_success"])
    annotation = row.get("annotation") or {}
    if annotation.get("success") is not None:
        return bool(annotation["success"])
    failure_type = str(row.get("failure_type") or "").lower()
    outcome = str(row.get("outcome") or "").lower()
    return failure_type == "success" or outcome == "success"


def _episode_failure_type(row: dict[str, Any], success: bool) -> str:
    if success:
        return "success"
    return str(row.get("failure_type") or row.get("outcome") or "unknown_failure")


def _segment_frame_range(segment: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not isinstance(segment, dict):
        return None, None
    start = _as_optional_int(segment.get("start_frame_index"))
    end = _as_optional_int(segment.get("end_frame_index"))
    return start, end


def _row_segment_role(frame_index: int, episode_len: int, row: dict[str, Any]) -> str:
    success = _episode_success(row)
    if success and not (row.get("intervention") or {}).get("has_intervention"):
        return "success_policy"

    segments = row.get("segments") or {}
    recovery_start, recovery_end = _segment_frame_range(segments.get("recovery"))
    correction_start, correction_end = _segment_frame_range(segments.get("correction"))
    human_start, human_end = _segment_frame_range(segments.get("human_correction"))
    prefix_start, prefix_end = _segment_frame_range(segments.get("policy_prefix"))

    if recovery_start is not None and recovery_end is not None and recovery_start <= frame_index <= recovery_end:
        return "recovery"
    if (
        correction_start is not None
        and correction_end is not None
        and correction_start <= frame_index <= correction_end
    ):
        return "correction"
    if human_start is not None and human_end is not None and human_start <= frame_index <= human_end:
        return "human_correction"
    if prefix_start is not None and prefix_end is not None and prefix_start <= frame_index <= prefix_end:
        return "failed_policy_prefix"

    intervention = row.get("intervention") or {}
    start_frame = _as_optional_int(intervention.get("start_frame_index"))
    end_frame = _as_optional_int(intervention.get("end_frame_index"))
    recovery_done_frame = _as_optional_int(intervention.get("recovery_done_frame_index"))
    if start_frame is not None and frame_index < start_frame:
        return "failed_policy_prefix"
    if start_frame is not None and end_frame is not None and start_frame <= frame_index <= end_frame:
        if recovery_done_frame is not None and frame_index >= recovery_done_frame:
            return "correction"
        if recovery_done_frame is not None:
            return "recovery"
        return "human_correction"

    if success:
        return "success_policy"
    if intervention.get("has_intervention"):
        if start_frame is not None and frame_index >= start_frame:
            return "human_correction"
        return "failed_policy_prefix"

    if episode_len > 0:
        return "failed_policy_no_correction"
    return "unknown"


def _normalized_return_target(
    frame_index: int,
    episode_len: int,
    max_episode_len: int,
    *,
    terminal_success: bool,
    c_fail_coef: float,
) -> float:
    remaining_steps = max(0, int(episode_len) - int(frame_index) - 1)
    c_fail = float(max_episode_len) * float(c_fail_coef)
    raw_return = -float(remaining_steps)
    if not terminal_success:
        raw_return -= c_fail
    denom = float(max_episode_len) + c_fail
    if denom <= 0:
        return 0.0
    return float(np.clip(raw_return / denom, -1.0, 0.0))


def _state_to_list(value: Any) -> list[float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return arr.astype(float).tolist()


def _build_source_targets(
    source: SourceSpec,
    *,
    max_episode_len: int,
    c_fail_coef: float,
    segment_success: dict[str, bool],
    loss_weights: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _load_dataset_frames(source.dataset_root)
    info = _load_info(source.dataset_root)
    manifest_rows = _load_jsonl(source.manifest)
    manifest_by_episode = {int(row["episode_index"]): row for row in manifest_rows}

    output_rows: list[dict[str, Any]] = []
    missing_manifest_episodes: list[int] = []

    for episode_index, ep_df in df.groupby("episode_index", sort=True):
        ep_idx = int(episode_index)
        row = manifest_by_episode.get(ep_idx)
        if row is None:
            missing_manifest_episodes.append(ep_idx)
            row = {
                "episode_index": ep_idx,
                "episode_id": f"episode_{ep_idx:06d}",
                "handover_success": False,
                "failure_type": "missing_manifest",
                "collection_round": source.name,
                "source_policy_generation": "unknown",
                "source_policy_checkpoint": None,
                "rl_use": "missing_manifest",
                "episode_role": "unknown",
            }

        ep_len = int(len(ep_df))
        ep_success = _episode_success(row)
        failure_type = _episode_failure_type(row, ep_success)
        episode_id = str(row.get("episode_id") or f"episode_{ep_idx:06d}")
        collection_round = str(row.get("collection_round") or row.get("run_id") or source.name)
        source_policy_generation = str(row.get("source_policy_generation") or "unknown")
        source_policy_checkpoint = row.get("source_policy_checkpoint") or row.get("policy_path")
        rl_use = str(row.get("rl_use") or ("positive_success" if ep_success else "unknown_failure"))
        episode_role = str(row.get("episode_role") or ("success_policy_rollout" if ep_success else "unknown"))
        has_intervention = bool((row.get("intervention") or {}).get("has_intervention"))

        for _, record_dict in ep_df.iterrows():
            frame_index = int(record_dict["frame_index"])
            global_index = int(record_dict["index"])
            timestamp = float(record_dict["timestamp"])
            segment_role = _row_segment_role(frame_index, ep_len, row)
            segment_terminal_success = bool(segment_success.get(segment_role, False))

            value_policy_outcome = _normalized_return_target(
                frame_index,
                ep_len,
                max_episode_len,
                terminal_success=ep_success,
                c_fail_coef=c_fail_coef,
            )
            value_segment = _normalized_return_target(
                frame_index,
                ep_len,
                max_episode_len,
                terminal_success=segment_terminal_success,
                c_fail_coef=c_fail_coef,
            )
            frame_progress = float(frame_index / max(1, ep_len - 1))

            output_rows.append(
                {
                    "source_name": source.name,
                    "dataset_root": str(source.dataset_root),
                    "dataset_repo_id": row.get("dataset_repo_id") or info.get("repo_id"),
                    "episode_index": ep_idx,
                    "episode_id": episode_id,
                    "source_episode_uid": f"{source.name}:{ep_idx}",
                    "global_index": global_index,
                    "frame_index": frame_index,
                    "episode_length": ep_len,
                    "timestamp": timestamp,
                    "frame_progress": frame_progress,
                    "observation_state": _state_to_list(record_dict["observation.state"]),
                    "handover_success": ep_success,
                    "failure_type": failure_type,
                    "segment_role": segment_role,
                    "segment_terminal_success": segment_terminal_success,
                    "has_intervention": has_intervention,
                    "collection_round": collection_round,
                    "source_policy_generation": source_policy_generation,
                    "source_policy_checkpoint": source_policy_checkpoint,
                    "episode_role": episode_role,
                    "rl_use": rl_use,
                    "value_target_policy_outcome": value_policy_outcome,
                    "value_target_segment": value_segment,
                    "value_loss_weight": float(loss_weights.get(segment_role, loss_weights["unknown"])),
                }
            )

    targets = pd.DataFrame(output_rows)
    report = {
        "source": asdict(source),
        "dataset_rows": int(len(df)),
        "target_rows": int(len(targets)),
        "dataset_episodes": int(df["episode_index"].nunique()),
        "manifest_rows": int(len(manifest_rows)),
        "missing_manifest_episodes": missing_manifest_episodes,
        "fps": info.get("fps"),
        "features": info.get("features", {}),
    }
    return targets, report


def _summarize_targets(targets: pd.DataFrame) -> dict[str, Any]:
    group_cols = ["source_name", "segment_role"]
    segment_summary = (
        targets.groupby(group_cols)
        .agg(
            frame_count=("global_index", "count"),
            episode_count=("source_episode_uid", "nunique"),
            target_segment_mean=("value_target_segment", "mean"),
            target_policy_mean=("value_target_policy_outcome", "mean"),
            loss_weight_mean=("value_loss_weight", "mean"),
        )
        .reset_index()
    )
    outcome_summary = (
        targets.groupby(["source_name", "handover_success", "failure_type"])
        .agg(frame_count=("global_index", "count"), episode_count=("source_episode_uid", "nunique"))
        .reset_index()
    )
    return {
        "rows": int(len(targets)),
        "episodes": int(targets["source_episode_uid"].nunique()),
        "sources": sorted(targets["source_name"].unique().tolist()),
        "segment_summary": segment_summary.round(6).to_dict(orient="records"),
        "outcome_summary": outcome_summary.to_dict(orient="records"),
        "value_target_segment": {
            "min": float(targets["value_target_segment"].min()),
            "mean": float(targets["value_target_segment"].mean()),
            "max": float(targets["value_target_segment"].max()),
        },
        "value_target_policy_outcome": {
            "min": float(targets["value_target_policy_outcome"].min()),
            "mean": float(targets["value_target_policy_outcome"].mean()),
            "max": float(targets["value_target_policy_outcome"].max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Repeatable source spec: name=NAME::root=/path/to/dataset::manifest=/path/to/manifest.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="so101_handover_value_v0")
    parser.add_argument("--c-fail-coef", type=float, default=1.0)
    parser.add_argument("--max-episode-len", type=int, default=None)
    parser.add_argument("--segment-success-json", default=None)
    parser.add_argument("--loss-weights-json", default=None)
    args = parser.parse_args()

    sources = [_parse_key_value_source(raw) for raw in args.source]
    if args.c_fail_coef < 0:
        raise ValueError("--c-fail-coef must be non-negative")

    segment_success = dict(DEFAULT_SEGMENT_SUCCESS)
    if args.segment_success_json:
        segment_success.update(json.loads(args.segment_success_json))
    loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
    if args.loss_weights_json:
        loss_weights.update({k: float(v) for k, v in json.loads(args.loss_weights_json).items()})

    if args.max_episode_len is not None:
        max_episode_len = int(args.max_episode_len)
    else:
        lengths = []
        for source in sources:
            df = _load_dataset_frames(source.dataset_root)
            lengths.extend(df.groupby("episode_index").size().astype(int).tolist())
        max_episode_len = max(lengths)

    all_targets: list[pd.DataFrame] = []
    source_reports: list[dict[str, Any]] = []
    for source in sources:
        targets, report = _build_source_targets(
            source,
            max_episode_len=max_episode_len,
            c_fail_coef=args.c_fail_coef,
            segment_success=segment_success,
            loss_weights=loss_weights,
        )
        all_targets.append(targets)
        source_reports.append(report)

    targets_df = pd.concat(all_targets, ignore_index=True)

    out_dir = args.output_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    targets_path = out_dir / "value_targets_v0.parquet"
    report_path = out_dir / "value_targets_v0_report.json"
    lineage_path = out_dir / "lineage.json"

    targets_df.to_parquet(targets_path, index=False)
    report = {
        "schema_version": "so101_handover_value_targets.v0",
        "run_id": args.run_id,
        "target_path": str(targets_path),
        "c_fail_coef": args.c_fail_coef,
        "max_episode_len": max_episode_len,
        "segment_success": segment_success,
        "loss_weights": loss_weights,
        "summary": _summarize_targets(targets_df),
        "source_reports": source_reports,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    lineage_path.write_text(
        json.dumps(
            {
                "schema_version": "so101_handover_value_lineage.v0",
                "run_id": args.run_id,
                "sources": [asdict(source) for source in sources],
                "target_builder": str(Path(__file__).resolve()),
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {len(targets_df)} value target rows to {targets_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
