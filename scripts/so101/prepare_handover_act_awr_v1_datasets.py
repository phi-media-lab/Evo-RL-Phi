#!/usr/bin/env python
"""Prepare SO101 handover ACT-AWR v1 datasets from HIL rollout batches.

The script creates:

1. A batch2 RL-ready dataset from a raw MakerMods HIL eval dataset.
2. A combined batch1+batch2 RL-ready dataset with remapped episode/index/video
   references.

It intentionally writes new dataset roots and sidecar files instead of mutating
the original eval datasets.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_PARQUET = Path("data/chunk-000/file-000.parquet")
EPISODES_PARQUET = Path("meta/episodes/chunk-000/file-000.parquet")
INFO_JSON = Path("meta/info.json")
STATS_JSON = Path("meta/stats.json")
TASKS_PARQUET = Path("meta/tasks.parquet")
RUN_SIDECAR_FILES = [
    "hil_run.json",
    "annotations.json",
    "events.jsonl",
    "rl_manifest.json",
    "rl_manifest.jsonl",
    "record_action_log.jsonl",
]
BASE_DATA_COLUMNS = [
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "complementary_info.is_intervention",
]
VIDEO_FEATURES = [
    "observation.images.front_cam",
    "observation.images.hand_cam",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_clean_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def episode_id(index: int) -> str:
    return f"episode_{index:06d}"


def copy_file_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def load_rl_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "rl_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing rl_manifest.json. Export it first: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("episodes"), list):
        raise ValueError(f"Unexpected rl_manifest shape: {manifest_path}")
    return manifest


def build_phase_rows(
    manifest: dict[str, Any],
    *,
    run_id: str,
    source_dataset_repo_id: str,
    rlready_dataset_repo_id: str,
    base_policy_path: str,
    operator_confirmation: str,
    phase_failure_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    annotations: dict[str, dict[str, Any]] = {}

    for row in manifest["episodes"]:
        idx = int(row["episode_index"])
        ep_id = str(row.get("episode_id") or episode_id(idx))
        success = bool(row.get("handover_success"))
        failure_type = "success" if success else phase_failure_label
        reward = 1.0 if success else 0.0
        dataset_episode = row.get("dataset_episode") or {}
        intervention = row.get("intervention") or {}
        segments = row.get("segments") or {}
        policy_prefix = segments.get("policy_prefix") or {}
        recovery = segments.get("recovery") or {}
        correction = segments.get("correction") or {}
        human_correction = segments.get("human_correction") or {}
        has_intervention = bool(intervention.get("has_intervention"))
        rl_use = row.get(
            "rl_use",
            "positive_success"
            if success
            else "failed_policy_with_human_correction"
            if has_intervention
            else "failed_policy_no_correction",
        )

        annotation = (row.get("annotation") or {}).copy()
        annotation.update(
            {
                "episode_id": ep_id,
                "episode_index": idx,
                "success": success,
                "failure_type": failure_type,
                "handover_success": success,
            }
        )
        if not success:
            note = annotation.get("notes") or ""
            relabel_note = (
                f"Relabeled for ACT-AWR v1 on {now_iso()}: operator workflow treats this "
                f"intervention as {phase_failure_label}."
            )
            annotation["notes"] = f"{note} {relabel_note}".strip()
            annotation["phase_relabel_source"] = "operator_hil_batch2_assumption"
            annotation["phase_relabel_at"] = now_iso()
        annotations[ep_id] = annotation

        rows.append(
            {
                "run_id": run_id,
                "episode_id": ep_id,
                "episode_index": idx,
                "collection_round": row.get("collection_round", run_id),
                "source_policy_checkpoint": row.get("source_policy_checkpoint", row.get("policy_path", base_policy_path)),
                "source_policy_generation": row.get("source_policy_generation", Path(base_policy_path).name),
                "episode_role": row.get(
                    "episode_role",
                    "success_policy_rollout"
                    if success
                    else "failed_policy_with_intervention"
                    if has_intervention
                    else "failed_policy_no_correction",
                ),
                "dataset_from_index": dataset_episode.get("dataset_from_index"),
                "dataset_to_index": dataset_episode.get("dataset_to_index"),
                "length": dataset_episode.get("length"),
                "phase": "handover",
                "handover_success": success,
                "failure_type": failure_type,
                "handover_reward": reward,
                "policy_prefix": {
                    "start_frame_index": policy_prefix.get("start_frame_index", 0),
                    "end_frame_index": policy_prefix.get("end_frame_index"),
                    "frame_count": policy_prefix.get("frame_count"),
                },
                "segments": {
                    "policy_prefix": {
                        "start_frame_index": policy_prefix.get("start_frame_index", 0),
                        "end_frame_index": policy_prefix.get("end_frame_index"),
                        "frame_count": policy_prefix.get("frame_count"),
                    },
                    "recovery": {
                        "start_frame_index": recovery.get("start_frame_index"),
                        "end_frame_index": recovery.get("end_frame_index"),
                        "frame_count": recovery.get("frame_count"),
                    },
                    "correction": {
                        "start_frame_index": correction.get("start_frame_index"),
                        "end_frame_index": correction.get("end_frame_index"),
                        "frame_count": correction.get("frame_count"),
                    },
                    "human_correction": {
                        "start_frame_index": human_correction.get("start_frame_index"),
                        "end_frame_index": human_correction.get("end_frame_index"),
                        "frame_count": human_correction.get("frame_count"),
                    },
                },
                "human_correction": {
                    "start_frame_index": human_correction.get("start_frame_index"),
                    "end_frame_index": human_correction.get("end_frame_index"),
                    "frame_count": human_correction.get("frame_count"),
                },
                "intervention": {
                    "has_intervention": has_intervention,
                    "start_frame_index": intervention.get("start_frame_index"),
                    "recovery_done_frame_index": intervention.get("recovery_done_frame_index"),
                    "end_frame_index": intervention.get("end_frame_index"),
                    "correction_duration_s": intervention.get("correction_duration_s"),
                    "trigger": intervention.get("trigger"),
                    "phase": intervention.get("phase"),
                },
                "rl_use": rl_use,
            }
        )

    success_ids = [row["episode_id"] for row in rows if row["handover_success"]]
    failed_ids = [row["episode_id"] for row in rows if not row["handover_success"]]
    intervention_ids = [row["episode_id"] for row in rows if row["intervention"]["has_intervention"]]
    summary = {
        "schema_version": "so101_handover_rl_seed_phase_summary.v1",
        "generated_at": now_iso(),
        "run_id": run_id,
        "source_dataset_repo_id": source_dataset_repo_id,
        "rlready_dataset_repo_id": rlready_dataset_repo_id,
        "base_policy_path": base_policy_path,
        "task_focus": "handover",
        "operator_confirmation": operator_confirmation,
        "total_episodes": len(rows),
        "reached_handover_count": len(rows),
        "handover_success_count": len(success_ids),
        "handover_failed_count": len(failed_ids),
        "handover_success_rate": len(success_ids) / len(rows) if rows else 0.0,
        "intervention_count": len(intervention_ids),
        "failure_counts": dict(Counter(row["failure_type"] for row in rows)),
        "reward_schema": {
            "handover_success": 1.0,
            phase_failure_label: 0.0,
            "notes": "Reward is trajectory-level handover outcome; intervention episodes are failed policy rollouts with human correction segments available.",
        },
        "reward_counts": dict(Counter(str(row["handover_reward"]) for row in rows)),
        "handover_success_episodes": success_ids,
        "handover_failed_episodes": failed_ids,
        "intervention_episodes": intervention_ids,
        "training_recommendation": "Use as a hard-failure second HIL batch for handover-focused ACT-AWR.",
        "rl_status": "seed_ready_not_sufficient_as_final_dataset",
    }
    return summary, rows, annotations


def add_intervention_column(df: pd.DataFrame, phase_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = df.copy()
    intervention = np.zeros(len(df), dtype=np.float32)
    episode_indices = df["episode_index"].astype(int).to_numpy()
    frame_indices = df["frame_index"].astype(int).to_numpy()
    for row in phase_rows:
        start = row["human_correction"].get("start_frame_index")
        end = row["human_correction"].get("end_frame_index")
        if start is None or end is None:
            continue
        mask = (episode_indices == int(row["episode_index"])) & (frame_indices >= int(start)) & (frame_indices <= int(end))
        intervention[mask] = 1.0
    df["complementary_info.is_intervention"] = intervention
    return df


def patch_info_features(info: dict[str, Any]) -> dict[str, Any]:
    info = json.loads(json.dumps(info))
    features = info.setdefault("features", {})
    for field in ["complementary_info.value_batch1_smoke", "complementary_info.advantage_batch1_smoke", "complementary_info.acp_indicator_batch1_smoke"]:
        features.pop(field, None)
    features["complementary_info.is_intervention"] = {
        "dtype": "float32",
        "shape": [1],
        "names": None,
    }
    return info


def patch_episode_success(episodes_df: pd.DataFrame, phase_rows: list[dict[str, Any]]) -> pd.DataFrame:
    labels = {
        int(row["episode_index"]): "success" if row["handover_success"] else "failure"
        for row in phase_rows
    }
    episodes_df = episodes_df.copy()
    episodes_df["episode_success"] = [labels[int(idx)] for idx in episodes_df["episode_index"]]
    return episodes_df


def write_handover_sidecar(
    sidecar_dir: Path,
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    source_run_dir: Path,
    conversion_report: dict[str, Any],
) -> None:
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_SIDECAR_FILES:
        copy_file_if_exists(source_run_dir / name, sidecar_dir / name)
    write_json(sidecar_dir / "annotations.json", annotations)
    write_json(sidecar_dir / "phase_summary.json", summary)
    write_json(sidecar_dir / "handover_rl_seed_manifest_v0.json", {"schema_version": "so101_handover_rl_seed_manifest.v0", "generated_at": now_iso(), "summary": summary, "episodes": rows})
    write_jsonl(sidecar_dir / "handover_rl_seed_manifest_v0.jsonl", rows)
    write_json(sidecar_dir / "rl_ready_conversion_report.json", conversion_report)


def build_batch2(args: argparse.Namespace) -> dict[str, Any]:
    src_root = args.batch2_source_root.expanduser().resolve()
    dst_root = args.batch2_output_root.expanduser().resolve()
    run_dir = args.batch2_run_dir.expanduser().resolve()
    run_id = run_dir.name
    ensure_clean_output(dst_root, args.overwrite)
    shutil.copytree(src_root, dst_root)

    manifest = load_rl_manifest(run_dir)
    summary, rows, annotations = build_phase_rows(
        manifest,
        run_id=run_id,
        source_dataset_repo_id=args.batch2_source_repo_id,
        rlready_dataset_repo_id=args.batch2_output_repo_id,
        base_policy_path=args.batch2_base_policy_path,
        operator_confirmation=args.batch2_operator_confirmation,
        phase_failure_label=args.phase_failure_label,
    )

    df = pd.read_parquet(dst_root / DATA_PARQUET)
    df = add_intervention_column(df, rows)
    df.to_parquet(dst_root / DATA_PARQUET, index=False)

    episodes_df = pd.read_parquet(dst_root / EPISODES_PARQUET)
    episodes_df = patch_episode_success(episodes_df, rows)
    episodes_df.to_parquet(dst_root / EPISODES_PARQUET, index=False)

    info = patch_info_features(read_json(dst_root / INFO_JSON))
    info["total_frames"] = int(len(df))
    info["total_episodes"] = int(episodes_df["episode_index"].nunique())
    write_json(dst_root / INFO_JSON, info)

    sidecar_dir = dst_root / "makermods_hil" / run_id
    conversion_report = {
        "source_dataset_repo_id": args.batch2_source_repo_id,
        "rl_ready_dataset_repo_id": args.batch2_output_repo_id,
        "dataset_root": str(dst_root),
        "episode_count": len(rows),
        "success_count": summary["handover_success_count"],
        "failure_count": summary["handover_failed_count"],
        "episodes_with_intervention": summary["intervention_count"],
        "intervention_frames": int(df["complementary_info.is_intervention"].sum()),
        "intervention_ranges": {
            str(row["episode_index"]): [
                row["human_correction"]["start_frame_index"],
                row["human_correction"]["end_frame_index"],
            ]
            for row in rows
            if row["intervention"]["has_intervention"]
        },
        "standard_fields_added": [
            "meta/episodes.episode_success",
            "data.complementary_info.is_intervention",
        ],
    }
    write_handover_sidecar(
        sidecar_dir,
        summary=summary,
        rows=rows,
        annotations=annotations,
        source_run_dir=run_dir,
        conversion_report=conversion_report,
    )
    return {"root": dst_root, "run_id": run_id, "summary": summary, "rows": rows}


def select_data_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in BASE_DATA_COLUMNS:
        if col not in result.columns:
            if col == "complementary_info.is_intervention":
                result[col] = np.zeros(len(result), dtype=np.float32)
            else:
                raise ValueError(f"Missing required data column: {col}")
    return result[BASE_DATA_COLUMNS]


def combine_numeric_stats(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for col in columns:
        arr = np.stack(df[col].to_numpy()) if isinstance(df[col].iloc[0], np.ndarray) else df[col].to_numpy()
        arr = arr.astype(float)
        stats[col] = {
            "min": np.nanmin(arr, axis=0).tolist() if arr.ndim > 1 else [float(np.nanmin(arr))],
            "max": np.nanmax(arr, axis=0).tolist() if arr.ndim > 1 else [float(np.nanmax(arr))],
            "mean": np.nanmean(arr, axis=0).tolist() if arr.ndim > 1 else [float(np.nanmean(arr))],
            "std": np.nanstd(arr, axis=0).tolist() if arr.ndim > 1 else [float(np.nanstd(arr))],
            "count": [float(len(df))],
        }
    return stats


def combine_video_stats(stat_a: dict[str, Any], stat_b: dict[str, Any]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for key in VIDEO_FEATURES:
        if key not in stat_a or key not in stat_b:
            continue
        a = stat_a[key]
        b = stat_b[key]
        ca = float(a.get("count", [0])[0] or 0)
        cb = float(b.get("count", [0])[0] or 0)
        total = max(ca + cb, 1.0)
        mean_a = np.asarray(a["mean"], dtype=float)
        mean_b = np.asarray(b["mean"], dtype=float)
        std_a = np.asarray(a["std"], dtype=float)
        std_b = np.asarray(b["std"], dtype=float)
        mean = (mean_a * ca + mean_b * cb) / total
        var = (ca * (std_a**2 + (mean_a - mean) ** 2) + cb * (std_b**2 + (mean_b - mean) ** 2)) / total
        combined[key] = {
            "min": np.minimum(np.asarray(a["min"], dtype=float), np.asarray(b["min"], dtype=float)).tolist(),
            "max": np.maximum(np.asarray(a["max"], dtype=float), np.asarray(b["max"], dtype=float)).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "count": [total],
        }
    return combined


def copy_and_remap_videos(src_root: Path, dst_root: Path, episodes_df: pd.DataFrame, file_offsets: dict[str, int]) -> pd.DataFrame:
    episodes_df = episodes_df.copy()
    for feature in VIDEO_FEATURES:
        file_col = f"videos/{feature}/file_index"
        chunk_col = f"videos/{feature}/chunk_index"
        if file_col not in episodes_df:
            continue
        offset = file_offsets.get(feature, 0)
        old_to_new = {
            int(old): int(old) + offset for old in sorted(episodes_df[file_col].dropna().astype(int).unique())
        }
        for old, new in old_to_new.items():
            old_chunk = int(episodes_df.loc[episodes_df[file_col].astype(int) == old, chunk_col].iloc[0])
            src = src_root / "videos" / feature / f"chunk-{old_chunk:03d}" / f"file-{old:03d}.mp4"
            dst = dst_root / "videos" / feature / f"chunk-{old_chunk:03d}" / f"file-{new:03d}.mp4"
            copy_file_if_exists(src, dst)
        episodes_df[file_col] = episodes_df[file_col].map(lambda value: old_to_new[int(value)])
    return episodes_df


def offset_manifest_rows(rows: list[dict[str, Any]], episode_offset: int, frame_offset: int) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        new = json.loads(json.dumps(row))
        new_idx = int(new["episode_index"]) + episode_offset
        new["source_episode_index"] = int(row["episode_index"])
        new["episode_index"] = new_idx
        new["episode_id"] = episode_id(new_idx)
        for key in ["dataset_from_index", "dataset_to_index"]:
            if new.get(key) is not None:
                new[key] = int(new[key]) + frame_offset
        output.append(new)
    return output


def build_combined(args: argparse.Namespace, batch2_result: dict[str, Any] | None) -> dict[str, Any]:
    batch1_root = args.batch1_root.expanduser().resolve()
    batch2_root = (batch2_result["root"] if batch2_result else args.batch2_output_root.expanduser().resolve())
    dst_root = args.combined_output_root.expanduser().resolve()
    ensure_clean_output(dst_root, args.overwrite)
    for path in [dst_root / "data/chunk-000", dst_root / "meta/episodes/chunk-000", dst_root / "videos"]:
        path.mkdir(parents=True, exist_ok=True)

    b1_df = select_data_columns(pd.read_parquet(batch1_root / DATA_PARQUET))
    b2_df = select_data_columns(pd.read_parquet(batch2_root / DATA_PARQUET))
    frame_offset = int(len(b1_df))
    episode_offset = int(pd.read_parquet(batch1_root / EPISODES_PARQUET)["episode_index"].max()) + 1

    b2_df = b2_df.copy()
    b2_df["index"] = b2_df["index"].astype(int) + frame_offset
    b2_df["episode_index"] = b2_df["episode_index"].astype(int) + episode_offset
    combined_df = pd.concat([b1_df, b2_df], ignore_index=True)
    combined_df.to_parquet(dst_root / DATA_PARQUET, index=False)

    b1_ep = pd.read_parquet(batch1_root / EPISODES_PARQUET)
    b2_ep = pd.read_parquet(batch2_root / EPISODES_PARQUET).copy()
    b2_ep["episode_index"] = b2_ep["episode_index"].astype(int) + episode_offset
    b2_ep["dataset_from_index"] = b2_ep["dataset_from_index"].astype(int) + frame_offset
    b2_ep["dataset_to_index"] = b2_ep["dataset_to_index"].astype(int) + frame_offset

    shutil.copy2(batch1_root / TASKS_PARQUET, dst_root / TASKS_PARQUET)
    for feature in VIDEO_FEATURES:
        src_dir = batch1_root / "videos" / feature / "chunk-000"
        dst_dir = dst_root / "videos" / feature / "chunk-000"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for file in src_dir.glob("file-*.mp4"):
            shutil.copy2(file, dst_dir / file.name)

    file_offsets = {}
    for feature in VIDEO_FEATURES:
        file_col = f"videos/{feature}/file_index"
        file_offsets[feature] = int(b1_ep[file_col].max()) + 1
    b2_ep = copy_and_remap_videos(batch2_root, dst_root, b2_ep, file_offsets)
    combined_ep = pd.concat([b1_ep, b2_ep], ignore_index=True)
    combined_ep.to_parquet(dst_root / EPISODES_PARQUET, index=False)

    info = patch_info_features(read_json(batch1_root / INFO_JSON))
    info["total_episodes"] = int(combined_ep["episode_index"].nunique())
    info["total_frames"] = int(len(combined_df))
    info["splits"] = {"train": f"0:{len(combined_df)}"}
    write_json(dst_root / INFO_JSON, info)

    stats = combine_numeric_stats(
        combined_df,
        ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"],
    )
    stats["complementary_info.is_intervention"] = combine_numeric_stats(
        combined_df,
        ["complementary_info.is_intervention"],
    )["complementary_info.is_intervention"]
    stats.update(combine_video_stats(read_json(batch1_root / STATS_JSON), read_json(batch2_root / STATS_JSON)))
    write_json(dst_root / STATS_JSON, stats)

    run_id = args.combined_run_id
    sidecar_dir = dst_root / "makermods_hil" / run_id
    b1_sidecar = batch1_root / "makermods_hil" / args.batch1_run_id
    b2_sidecar = batch2_root / "makermods_hil" / args.batch2_run_id
    b1_rows = read_jsonl(b1_sidecar / "handover_rl_seed_manifest_v0.jsonl")
    b2_rows = read_jsonl(b2_sidecar / "handover_rl_seed_manifest_v0.jsonl")
    combined_rows = offset_manifest_rows(b1_rows, 0, 0) + offset_manifest_rows(b2_rows, episode_offset, frame_offset)

    success_ids = [row["episode_id"] for row in combined_rows if row["handover_success"]]
    failed_ids = [row["episode_id"] for row in combined_rows if not row["handover_success"]]
    intervention_ids = [row["episode_id"] for row in combined_rows if row["intervention"]["has_intervention"]]
    summary = {
        "schema_version": "so101_handover_rl_seed_phase_summary.v1",
        "generated_at": now_iso(),
        "run_id": run_id,
        "source_dataset_repo_id": f"{args.batch1_repo_id}+{args.batch2_output_repo_id}",
        "rlready_dataset_repo_id": args.combined_output_repo_id,
        "base_policy_path": args.combined_base_policy_path,
        "task_focus": "handover",
        "operator_confirmation": "Combined ACT-AWR v1 training pool from batch1 baseline HIL seed and batch2 ACT-AWR v0 hard failures.",
        "total_episodes": len(combined_rows),
        "reached_handover_count": len(combined_rows),
        "handover_success_count": len(success_ids),
        "handover_failed_count": len(failed_ids),
        "handover_success_rate": len(success_ids) / len(combined_rows) if combined_rows else 0.0,
        "intervention_count": len(intervention_ids),
        "failure_counts": dict(Counter(row["failure_type"] for row in combined_rows)),
        "reward_schema": {
            "handover_success": 1.0,
            args.phase_failure_label: 0.0,
            "notes": "Combined handover-focused ACT-AWR v1 pool.",
        },
        "reward_counts": dict(Counter(str(row["handover_reward"]) for row in combined_rows)),
        "handover_success_episodes": success_ids,
        "handover_failed_episodes": failed_ids,
        "intervention_episodes": intervention_ids,
        "training_recommendation": "Train ACT-AWR v1 from the current ACT-AWR v0 10000-step checkpoint.",
        "rl_status": "combined_training_pool_ready",
    }
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    write_json(sidecar_dir / "phase_summary.json", summary)
    write_json(sidecar_dir / "handover_rl_seed_manifest_v0.json", {"schema_version": "so101_handover_rl_seed_manifest.v0", "generated_at": now_iso(), "summary": summary, "episodes": combined_rows})
    write_jsonl(sidecar_dir / "handover_rl_seed_manifest_v0.jsonl", combined_rows)
    write_json(
        sidecar_dir / "combined_dataset_report.json",
        {
            "combined_root": str(dst_root),
            "batch1_root": str(batch1_root),
            "batch2_root": str(batch2_root),
            "episode_offset": episode_offset,
            "frame_offset": frame_offset,
            "total_episodes": int(combined_ep["episode_index"].nunique()),
            "total_frames": int(len(combined_df)),
            "video_file_offsets": file_offsets,
            "summary": summary,
        },
    )
    return {"root": dst_root, "run_id": run_id, "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SO101 handover ACT-AWR v1 RL-ready datasets.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--phase-failure-label", default="handover_failed")

    parser.add_argument("--batch1-root", type=Path, required=True)
    parser.add_argument("--batch1-repo-id", required=True)
    parser.add_argument("--batch1-run-id", required=True)

    parser.add_argument("--batch2-source-root", type=Path, required=True)
    parser.add_argument("--batch2-source-repo-id", required=True)
    parser.add_argument("--batch2-output-root", type=Path, required=True)
    parser.add_argument("--batch2-output-repo-id", required=True)
    parser.add_argument("--batch2-run-dir", type=Path, required=True)
    parser.add_argument("--batch2-run-id", required=True)
    parser.add_argument("--batch2-base-policy-path", required=True)
    parser.add_argument(
        "--batch2-operator-confirmation",
        default="The intervention failures in batch2 are treated as handover-phase failures for ACT-AWR v1 training.",
    )

    parser.add_argument("--combined-output-root", type=Path, required=True)
    parser.add_argument("--combined-output-repo-id", required=True)
    parser.add_argument("--combined-run-id", required=True)
    parser.add_argument("--combined-base-policy-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch2 = build_batch2(args)
    combined = build_combined(args, batch2)
    print("Batch2 RL-ready dataset:")
    print(json.dumps({"root": str(batch2["root"]), "run_id": batch2["run_id"], "summary": batch2["summary"]}, ensure_ascii=False, indent=2))
    print("Combined RL-ready dataset:")
    print(json.dumps({"root": str(combined["root"]), "run_id": combined["run_id"], "summary": combined["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
