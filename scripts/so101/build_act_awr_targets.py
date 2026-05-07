#!/usr/bin/env python
"""Build ACT-AWR/AWAC bootstrap targets from SO101 HIL sidecar metadata.

This script intentionally writes a sidecar parquet instead of mutating the
LeRobot dataset schema. The ACT trainer can later join the output by the
global ``index`` column and use ``act_awr_weight`` for per-sample loss weights.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_PARQUET = "data/chunk-000/file-000.parquet"
DEFAULT_MANIFEST_JSONL = "handover_rl_seed_manifest_v0.jsonl"
DEFAULT_MANIFEST_JSON = "handover_rl_seed_manifest_v0.json"
RAC_V2_SEGMENT_MULTIPLIERS = {
    "recovery": 3.0,
    "correction": 3.0,
    "human_correction": 2.0,
    "success_policy": 1.0,
    "failed_policy_prefix": 0.05,
    "failed_policy_suffix": 0.05,
    "failed_policy_no_correction": 0.0,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return rows


def _load_manifest(sidecar_dir: Path) -> list[dict[str, Any]]:
    jsonl_path = sidecar_dir / DEFAULT_MANIFEST_JSONL
    if jsonl_path.exists():
        return _read_jsonl(jsonl_path)

    json_path = sidecar_dir / DEFAULT_MANIFEST_JSON
    if json_path.exists():
        obj = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("episodes"), list):
            return obj["episodes"]
        raise ValueError(f"Unsupported manifest JSON shape in {json_path}")

    rl_manifest_path = sidecar_dir / "rl_manifest.json"
    if rl_manifest_path.exists():
        obj = json.loads(rl_manifest_path.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("episodes"), list):
            return obj["episodes"]
        raise ValueError(f"Unsupported rl_manifest JSON shape in {rl_manifest_path}")

    raise FileNotFoundError(
        f"No manifest found in {sidecar_dir}; expected {DEFAULT_MANIFEST_JSONL}, "
        f"{DEFAULT_MANIFEST_JSON}, or rl_manifest.json"
    )


def _none_if_nan(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _as_optional_int(value: Any) -> int | None:
    value = _none_if_nan(value)
    if value is None:
        return None
    return int(value)


def _as_optional_float(value: Any) -> float | None:
    value = _none_if_nan(value)
    if value is None:
        return None
    return float(value)


def _load_float_mapping(value: str | None, *, option_name: str) -> dict[str, float]:
    if not value:
        return {}
    try:
        obj = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} must be a JSON object mapping string keys to numeric values") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{option_name} must be a JSON object")
    mapping: dict[str, float] = {}
    for key, item in obj.items():
        try:
            mapping[str(key)] = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{option_name}[{key!r}] must be numeric") from exc
    return mapping


def _segment_bounds(segment: dict[str, Any]) -> tuple[int | None, int | None]:
    return (
        _as_optional_int(segment.get("start_frame_index")),
        _as_optional_int(segment.get("end_frame_index")),
    )


def _episode_id_from_index(episode_index: int) -> str:
    return f"episode_{episode_index:06d}"


def _manifest_by_episode(manifest_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    for row in manifest_rows:
        if "episode_index" not in row:
            continue
        episode_index = int(row["episode_index"])
        if episode_index in mapping:
            raise ValueError(f"Duplicate episode_index={episode_index} in manifest")
        mapping[episode_index] = row
    return mapping


def _baseline_value(mode: str, frame_returns: np.ndarray, episode_success_rate: float) -> float:
    if mode == "frame_return_mean":
        return float(np.nanmean(frame_returns))
    if mode == "episode_success_rate":
        return float(episode_success_rate)
    if mode == "zero":
        return 0.0
    raise ValueError(f"Unsupported baseline mode: {mode}")


def _compute_chunk_stat(
    values: np.ndarray,
    frame_indices: np.ndarray,
    chunk_size: int,
    reducer: str,
) -> tuple[np.ndarray, np.ndarray]:
    if reducer not in {"mean", "max"}:
        raise ValueError(f"Unsupported reducer: {reducer}")

    n = len(values)
    chunk_values = np.empty(n, dtype=np.float32)
    end_frame_indices = np.empty(n, dtype=np.int64)

    for i in range(n):
        end = min(i + chunk_size, n)
        window = values[i:end]
        if reducer == "mean":
            chunk_values[i] = float(np.nanmean(window))
        else:
            chunk_values[i] = float(np.nanmax(window))
        end_frame_indices[i] = int(frame_indices[end - 1])

    return chunk_values, end_frame_indices


def build_targets(
    dataset_root: Path,
    run_id: str,
    data_parquet: str,
    output_dir: Path | None,
    chunk_size: int,
    beta: float,
    weight_min: float,
    weight_max: float,
    baseline_mode: str,
    success_return: float,
    correction_return: float,
    failed_prefix_return: float,
    failed_suffix_return: float,
    normalize_weights: bool,
    chunk_weight_mode: str,
    segment_multipliers: dict[str, float] | None = None,
    round_multipliers: dict[str, float] | None = None,
    current_round: str | None = None,
    current_round_multiplier: float = 1.0,
) -> tuple[Path, Path, dict[str, Any]]:
    if beta <= 0:
        raise ValueError("--beta must be positive")
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if weight_min <= 0:
        raise ValueError("--weight-min must be positive")
    if weight_max < weight_min:
        raise ValueError("--weight-max must be >= --weight-min")
    segment_multipliers = segment_multipliers or {}
    round_multipliers = round_multipliers or {}

    dataset_root = dataset_root.expanduser().resolve()
    sidecar_dir = output_dir.expanduser().resolve() if output_dir else dataset_root / "makermods_hil" / run_id
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    data_path = dataset_root / data_parquet
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset parquet not found: {data_path}")

    manifest_rows = _load_manifest(sidecar_dir)
    manifest = _manifest_by_episode(manifest_rows)
    df = pd.read_parquet(data_path)

    required_cols = {"index", "episode_index", "frame_index"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Dataset parquet missing required columns: {missing}")

    if "complementary_info.is_intervention" in df.columns:
        is_intervention = df["complementary_info.is_intervention"].fillna(0).astype(float).to_numpy() > 0
    else:
        is_intervention = np.zeros(len(df), dtype=bool)

    episode_indices = df["episode_index"].astype(int).to_numpy()
    frame_indices = df["frame_index"].astype(int).to_numpy()
    global_indices = df["index"].astype(int).to_numpy()

    segment_role = np.empty(len(df), dtype=object)
    handover_success = np.zeros(len(df), dtype=bool)
    failure_type = np.empty(len(df), dtype=object)
    episode_handover_reward = np.zeros(len(df), dtype=np.float32)
    return_to_go = np.zeros(len(df), dtype=np.float32)
    is_human_correction = np.zeros(len(df), dtype=bool)
    is_policy_prefix = np.zeros(len(df), dtype=bool)
    is_success_episode = np.zeros(len(df), dtype=bool)
    is_failed_episode = np.zeros(len(df), dtype=bool)
    human_start_frame = np.full(len(df), -1, dtype=np.int64)
    human_end_frame = np.full(len(df), -1, dtype=np.int64)
    episode_id_values = np.empty(len(df), dtype=object)
    collection_round_values = np.empty(len(df), dtype=object)
    source_policy_generation_values = np.empty(len(df), dtype=object)
    source_policy_checkpoint_values = np.empty(len(df), dtype=object)
    episode_role_values = np.empty(len(df), dtype=object)
    rl_use_values = np.empty(len(df), dtype=object)

    observed_episode_indices = sorted(int(x) for x in pd.unique(df["episode_index"]))
    missing_manifest_episodes = [idx for idx in observed_episode_indices if idx not in manifest]
    if missing_manifest_episodes:
        raise ValueError(f"Manifest missing episode indices: {missing_manifest_episodes}")

    episode_success_flags = []
    for episode_index in observed_episode_indices:
        row = manifest[episode_index]
        ep_mask = episode_indices == episode_index
        ep_frame_indices = frame_indices[ep_mask]
        ep_intervention = is_intervention[ep_mask]

        ep_success = bool(row.get("handover_success", row.get("success", False)))
        ep_failure_type = str(row.get("failure_type") or ("success" if ep_success else "unknown_failure"))
        ep_reward = float(row.get("handover_reward", 1.0 if ep_success else 0.0))
        episode_success_flags.append(ep_success)

        segments = row.get("segments") or {}
        intervention = row.get("intervention") or {}
        legacy_correction = row.get("human_correction") or segments.get("human_correction") or {}
        recovery = segments.get("recovery") or row.get("recovery") or {}
        correction = segments.get("correction") or row.get("correction") or {}
        has_explicit_recovery_correction = bool(recovery) or bool(correction)

        recovery_start, recovery_end = _segment_bounds(recovery)
        correction_start, correction_end = _segment_bounds(correction)
        legacy_correction_start, legacy_correction_end = _segment_bounds(legacy_correction)
        if correction_start is None:
            correction_start = legacy_correction_start
        if correction_end is None:
            correction_end = legacy_correction_end
        if correction_start is None:
            correction_start = _as_optional_int(intervention.get("start_frame_index"))
        if correction_end is None:
            correction_end = _as_optional_int(intervention.get("end_frame_index"))
        human_start = recovery_start if recovery_start is not None else correction_start
        human_end = correction_end

        if human_start is None and ep_intervention.any():
            human_start = int(ep_frame_indices[np.argmax(ep_intervention)])
        if correction_start is None:
            correction_start = human_start
        if human_end is None and human_start is not None:
            human_end = int(ep_frame_indices.max())
        if correction_end is None:
            correction_end = human_end

        ep_episode_id = str(row.get("episode_id") or _episode_id_from_index(episode_index))
        ep_collection_round = str(row.get("collection_round") or run_id)
        ep_source_policy_generation = str(row.get("source_policy_generation") or "unknown_policy")
        ep_source_policy_checkpoint = str(row.get("source_policy_checkpoint") or row.get("policy_path") or "")
        ep_episode_role = str(row.get("episode_role") or ("success_policy_rollout" if ep_success else "failed_policy_with_intervention" if human_start is not None else "failed_policy_no_correction"))
        ep_rl_use = str(row.get("rl_use") or ("positive_success" if ep_success else "failed_policy_with_human_correction" if human_start is not None else "failed_policy_no_correction"))
        episode_id_values[ep_mask] = ep_episode_id
        collection_round_values[ep_mask] = ep_collection_round
        source_policy_generation_values[ep_mask] = ep_source_policy_generation
        source_policy_checkpoint_values[ep_mask] = ep_source_policy_checkpoint
        episode_role_values[ep_mask] = ep_episode_role
        rl_use_values[ep_mask] = ep_rl_use
        handover_success[ep_mask] = ep_success
        failure_type[ep_mask] = ep_failure_type
        episode_handover_reward[ep_mask] = ep_reward
        is_success_episode[ep_mask] = ep_success
        is_failed_episode[ep_mask] = not ep_success
        if human_start is not None:
            human_start_frame[ep_mask] = human_start
        if human_end is not None:
            human_end_frame[ep_mask] = human_end

        ep_roles = np.empty(ep_mask.sum(), dtype=object)
        ep_returns = np.empty(ep_mask.sum(), dtype=np.float32)
        ep_is_human = np.zeros(ep_mask.sum(), dtype=bool)
        ep_is_prefix = np.zeros(ep_mask.sum(), dtype=bool)

        if ep_success:
            ep_roles[:] = "success_policy"
            ep_returns[:] = success_return
        elif human_start is not None:
            prefix_mask = ep_frame_indices < human_start
            recovery_mask = np.zeros(ep_mask.sum(), dtype=bool)
            correction_mask = np.zeros(ep_mask.sum(), dtype=bool)
            human_mask = np.zeros(ep_mask.sum(), dtype=bool)

            if recovery_start is not None:
                recovery_mask = ep_frame_indices >= recovery_start
                if recovery_end is not None:
                    recovery_mask &= ep_frame_indices <= recovery_end

            if correction_start is not None:
                correction_mask = ep_frame_indices >= correction_start
                if correction_end is not None:
                    correction_mask &= ep_frame_indices <= correction_end

            if has_explicit_recovery_correction and (recovery_mask.any() or correction_mask.any()):
                human_mask = recovery_mask | correction_mask
            else:
                human_mask = ep_frame_indices >= human_start
                if human_end is not None:
                    human_mask &= ep_frame_indices <= human_end
                human_mask |= ep_intervention

            suffix_mask = ~(human_mask | prefix_mask)

            ep_roles[prefix_mask] = "failed_policy_prefix"
            ep_returns[prefix_mask] = failed_prefix_return
            ep_is_prefix[prefix_mask] = True

            if has_explicit_recovery_correction and (recovery_mask.any() or correction_mask.any()):
                ep_roles[recovery_mask] = "recovery"
                ep_returns[recovery_mask] = correction_return
                ep_is_human[recovery_mask] = True

                ep_roles[correction_mask] = "correction"
                ep_returns[correction_mask] = correction_return
                ep_is_human[correction_mask] = True
            else:
                ep_roles[human_mask] = "human_correction"
                ep_returns[human_mask] = correction_return
                ep_is_human[human_mask] = True

            ep_roles[suffix_mask] = "failed_policy_suffix"
            ep_returns[suffix_mask] = failed_suffix_return
        else:
            ep_roles[:] = "failed_policy_no_correction"
            ep_returns[:] = failed_prefix_return
            ep_is_prefix[:] = True

        segment_role[ep_mask] = ep_roles
        return_to_go[ep_mask] = ep_returns
        is_human_correction[ep_mask] = ep_is_human
        is_policy_prefix[ep_mask] = ep_is_prefix

    episode_success_rate = float(np.mean(episode_success_flags)) if episode_success_flags else float("nan")
    baseline = _baseline_value(baseline_mode, return_to_go, episode_success_rate)
    advantage = return_to_go.astype(np.float32) - np.float32(baseline)
    awr_weight_raw = np.exp(advantage / np.float32(beta)).astype(np.float32)
    awr_weight_clipped = np.clip(awr_weight_raw, weight_min, weight_max).astype(np.float32)
    if normalize_weights:
        mean_weight = float(np.nanmean(awr_weight_clipped))
        awr_weight = (awr_weight_clipped / max(mean_weight, 1e-8)).astype(np.float32)
    else:
        mean_weight = float(np.nanmean(awr_weight_clipped))
        awr_weight = awr_weight_clipped

    segment_multiplier_values = np.array(
        [float(segment_multipliers.get(str(role), 1.0)) for role in segment_role],
        dtype=np.float32,
    )
    round_multiplier_values = np.array(
        [
            float(
                round_multipliers.get(
                    str(collection_round),
                    current_round_multiplier if current_round and str(collection_round) == current_round else 1.0,
                )
            )
            for collection_round in collection_round_values
        ],
        dtype=np.float32,
    )
    role_adjusted_weight_raw = (awr_weight * segment_multiplier_values * round_multiplier_values).astype(np.float32)
    if normalize_weights:
        adjusted_mean_weight = float(np.nanmean(role_adjusted_weight_raw))
        final_awr_weight = (role_adjusted_weight_raw / max(adjusted_mean_weight, 1e-8)).astype(np.float32)
    else:
        adjusted_mean_weight = float(np.nanmean(role_adjusted_weight_raw))
        final_awr_weight = role_adjusted_weight_raw

    chunk_mean = np.empty(len(df), dtype=np.float32)
    chunk_max = np.empty(len(df), dtype=np.float32)
    chunk_end_frame_index = np.empty(len(df), dtype=np.int64)
    for episode_index in observed_episode_indices:
        ep_mask = episode_indices == episode_index
        ep_positions = np.flatnonzero(ep_mask)
        ep_weights = final_awr_weight[ep_positions]
        ep_frames = frame_indices[ep_positions]
        ep_chunk_mean, ep_end = _compute_chunk_stat(ep_weights, ep_frames, chunk_size, reducer="mean")
        ep_chunk_max, _ = _compute_chunk_stat(ep_weights, ep_frames, chunk_size, reducer="max")
        chunk_mean[ep_positions] = ep_chunk_mean
        chunk_max[ep_positions] = ep_chunk_max
        chunk_end_frame_index[ep_positions] = ep_end

    if chunk_weight_mode == "max":
        act_awr_weight = chunk_max
    elif chunk_weight_mode == "mean":
        act_awr_weight = chunk_mean
    elif chunk_weight_mode == "current":
        act_awr_weight = awr_weight
    else:
        raise ValueError(f"Unsupported chunk weight mode: {chunk_weight_mode}")

    targets = pd.DataFrame(
        {
            "index": global_indices,
            "episode_index": episode_indices,
            "episode_id": episode_id_values,
            "frame_index": frame_indices,
            "collection_round": collection_round_values,
            "source_policy_generation": source_policy_generation_values,
            "source_policy_checkpoint": source_policy_checkpoint_values,
            "episode_role": episode_role_values,
            "rl_use": rl_use_values,
            "chunk_size": np.full(len(df), chunk_size, dtype=np.int64),
            "chunk_end_frame_index": chunk_end_frame_index,
            "segment_role": segment_role,
            "handover_success": handover_success,
            "failure_type": failure_type,
            "handover_reward": episode_handover_reward,
            "return_to_go": return_to_go,
            "value_baseline": np.full(len(df), baseline, dtype=np.float32),
            "advantage": advantage.astype(np.float32),
            "awr_weight_raw": awr_weight_raw,
            "awr_weight_clipped": awr_weight_clipped,
            "awr_weight_base": awr_weight,
            "segment_multiplier": segment_multiplier_values,
            "round_multiplier": round_multiplier_values,
            "role_adjusted_awr_weight_raw": role_adjusted_weight_raw,
            "awr_weight_normalized": final_awr_weight,
            "awr_weight": final_awr_weight,
            "chunk_awr_weight_mean": chunk_mean,
            "chunk_awr_weight_max": chunk_max,
            "act_awr_weight": act_awr_weight.astype(np.float32),
            "is_intervention": is_intervention,
            "is_policy_prefix": is_policy_prefix,
            "is_human_correction": is_human_correction,
            "is_success_episode": is_success_episode,
            "is_failed_episode": is_failed_episode,
            "human_correction_start_frame_index": human_start_frame,
            "human_correction_end_frame_index": human_end_frame,
        }
    )

    target_path = sidecar_dir / "act_awr_targets_v0.parquet"
    report_path = sidecar_dir / "act_awr_targets_v0_report.json"
    targets.to_parquet(target_path, index=False)

    segment_counts = Counter(str(x) for x in targets["segment_role"])
    failure_counts = Counter(str(x) for x in targets.drop_duplicates("episode_index")["failure_type"])
    segment_weight_means = (
        targets.groupby("segment_role")[["awr_weight", "act_awr_weight", "return_to_go", "advantage", "segment_multiplier", "round_multiplier"]]
        .mean()
        .round(6)
        .to_dict(orient="index")
    )
    gradient_mass = (
        targets.groupby(["collection_round", "segment_role"])
        .agg(
            frame_count=("index", "count"),
            mean_act_awr_weight=("act_awr_weight", "mean"),
            sum_act_awr_weight=("act_awr_weight", "sum"),
        )
        .reset_index()
    )
    total_gradient_mass = float(gradient_mass["sum_act_awr_weight"].sum()) if len(gradient_mass) else 0.0
    gradient_mass["gradient_mass_ratio"] = (
        gradient_mass["sum_act_awr_weight"] / max(total_gradient_mass, 1e-8)
    )
    episode_table = (
        targets.groupby("episode_index")
        .agg(
            episode_id=("episode_id", "first"),
            collection_round=("collection_round", "first"),
            source_policy_generation=("source_policy_generation", "first"),
            episode_role=("episode_role", "first"),
            rl_use=("rl_use", "first"),
            length=("index", "count"),
            handover_success=("handover_success", "first"),
            failure_type=("failure_type", "first"),
            handover_reward=("handover_reward", "first"),
            human_correction_start_frame_index=("human_correction_start_frame_index", "first"),
            human_correction_end_frame_index=("human_correction_end_frame_index", "first"),
            intervention_frames=("is_intervention", "sum"),
            human_correction_frames=("is_human_correction", "sum"),
            failed_policy_prefix_frames=("is_policy_prefix", "sum"),
            act_awr_weight_mean=("act_awr_weight", "mean"),
            act_awr_weight_max=("act_awr_weight", "max"),
        )
        .reset_index()
    )

    human_segment_mean = max(
        (
            segment_weight_means.get(role, {}).get("awr_weight", -1)
            for role in ("human_correction", "recovery", "correction")
        ),
        default=-1,
    )

    report = {
        "schema_version": "so101_act_awr_targets.v0",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "algorithm": "ACT-AWR bootstrap targets",
        "notes": [
            "This is a sidecar target file; the original LeRobot parquet schema is not mutated.",
            "v0 uses a constant reward-derived baseline, not a learned critic.",
            "The trainer should join by global index and use act_awr_weight for per-sample ACT loss.",
        ],
        "dataset_root": str(dataset_root),
        "run_id": run_id,
        "data_parquet": str(data_path),
        "sidecar_dir": str(sidecar_dir),
        "target_path": str(target_path),
        "report_path": str(report_path),
        "parameters": {
            "chunk_size": chunk_size,
            "beta": beta,
            "weight_min": weight_min,
            "weight_max": weight_max,
            "baseline_mode": baseline_mode,
            "success_return": success_return,
            "correction_return": correction_return,
            "failed_prefix_return": failed_prefix_return,
            "failed_suffix_return": failed_suffix_return,
            "normalize_weights": normalize_weights,
            "chunk_weight_mode": chunk_weight_mode,
            "segment_multipliers": segment_multipliers,
            "round_multipliers": round_multipliers,
            "current_round": current_round,
            "current_round_multiplier": current_round_multiplier,
        },
        "dataset_stats": {
            "num_rows": int(len(targets)),
            "num_episodes": int(len(observed_episode_indices)),
            "success_episodes": int(sum(episode_success_flags)),
            "failed_episodes": int(len(episode_success_flags) - sum(episode_success_flags)),
            "handover_success_rate": episode_success_rate,
            "failure_type_counts_by_episode": dict(failure_counts),
            "segment_counts_by_frame": dict(segment_counts),
            "frame_return_mean": float(np.nanmean(return_to_go)),
            "episode_success_rate_baseline": episode_success_rate,
            "value_baseline": baseline,
            "clipped_weight_mean_before_normalization": mean_weight,
            "adjusted_weight_mean_before_normalization": adjusted_mean_weight,
        },
        "weight_stats": {
            "awr_weight_raw": {
                "min": float(np.nanmin(awr_weight_raw)),
                "mean": float(np.nanmean(awr_weight_raw)),
                "max": float(np.nanmax(awr_weight_raw)),
            },
            "awr_weight": {
                "min": float(np.nanmin(awr_weight)),
                "mean": float(np.nanmean(awr_weight)),
                "max": float(np.nanmax(awr_weight)),
            },
            "act_awr_weight": {
                "min": float(np.nanmin(act_awr_weight)),
                "mean": float(np.nanmean(act_awr_weight)),
                "max": float(np.nanmax(act_awr_weight)),
            },
            "segment_means": segment_weight_means,
            "gradient_mass_by_round_and_segment": gradient_mass.round(6).to_dict(orient="records"),
            "total_gradient_mass": total_gradient_mass,
        },
        "acceptance_checks": {
            "weights_not_all_one": bool(np.nanmax(awr_weight) - np.nanmin(awr_weight) > 1e-6),
            "weights_not_all_max": bool(np.nanmean(awr_weight_clipped >= weight_max) < 0.99),
            "has_human_correction_frames": bool(segment_counts.get("human_correction", 0) > 0),
            "has_recovery_frames": bool(segment_counts.get("recovery", 0) > 0),
            "has_correction_frames": bool(segment_counts.get("correction", 0) > 0),
            "has_failed_policy_prefix_frames": bool(segment_counts.get("failed_policy_prefix", 0) > 0),
            "human_correction_mean_gt_failed_prefix_mean": bool(
                human_segment_mean
                > segment_weight_means.get("failed_policy_prefix", {}).get("awr_weight", float("inf"))
            ),
        },
        "episodes": episode_table.to_dict(orient="records"),
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    return target_path, report_path, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ACT-AWR bootstrap target weights for SO101 handover HIL datasets."
    )
    parser.add_argument("--dataset-root", required=True, type=Path, help="Local LeRobot dataset root.")
    parser.add_argument("--run-id", required=True, help="HIL run id under makermods_hil/.")
    parser.add_argument("--data-parquet", default=DEFAULT_DATA_PARQUET, help="Dataset parquet path under root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output sidecar dir. Defaults to <dataset-root>/makermods_hil/<run-id>.",
    )
    parser.add_argument("--chunk-size", type=int, default=100, help="ACT action chunk window size.")
    parser.add_argument("--beta", type=float, default=0.25, help="AWR temperature.")
    parser.add_argument("--weight-min", type=float, default=0.05, help="Minimum clipped frame weight.")
    parser.add_argument("--weight-max", type=float, default=5.0, help="Maximum clipped frame weight.")
    parser.add_argument(
        "--baseline-mode",
        choices=["frame_return_mean", "episode_success_rate", "zero"],
        default="frame_return_mean",
        help="Constant reward-derived v0 baseline. A learned V(s) should replace this in v1.",
    )
    parser.add_argument("--success-return", type=float, default=1.0)
    parser.add_argument("--correction-return", type=float, default=1.0)
    parser.add_argument("--failed-prefix-return", type=float, default=0.0)
    parser.add_argument("--failed-suffix-return", type=float, default=0.0)
    parser.add_argument(
        "--no-normalize-weights",
        action="store_true",
        help="Keep clipped AWR weights instead of normalizing them to mean 1.",
    )
    parser.add_argument(
        "--chunk-weight-mode",
        choices=["max", "mean", "current"],
        default="max",
        help="How to convert frame weights into ACT chunk sample weights.",
    )
    parser.add_argument(
        "--use-rac-v2-multipliers",
        action="store_true",
        help="Apply the default RaC-style segment multipliers for recovery/correction training.",
    )
    parser.add_argument(
        "--segment-multipliers-json",
        default=None,
        help="JSON object overriding segment role multipliers, e.g. '{\"recovery\":3,\"correction\":3}'.",
    )
    parser.add_argument(
        "--round-multipliers-json",
        default=None,
        help="JSON object mapping collection_round values to multipliers.",
    )
    parser.add_argument(
        "--current-round",
        default=None,
        help="Optional collection_round name that should receive --current-round-multiplier.",
    )
    parser.add_argument("--current-round-multiplier", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segment_multipliers = dict(RAC_V2_SEGMENT_MULTIPLIERS) if args.use_rac_v2_multipliers else {}
    segment_multipliers.update(
        _load_float_mapping(args.segment_multipliers_json, option_name="--segment-multipliers-json")
    )
    round_multipliers = _load_float_mapping(args.round_multipliers_json, option_name="--round-multipliers-json")
    target_path, report_path, report = build_targets(
        dataset_root=args.dataset_root,
        run_id=args.run_id,
        data_parquet=args.data_parquet,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        beta=args.beta,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        baseline_mode=args.baseline_mode,
        success_return=args.success_return,
        correction_return=args.correction_return,
        failed_prefix_return=args.failed_prefix_return,
        failed_suffix_return=args.failed_suffix_return,
        normalize_weights=not args.no_normalize_weights,
        chunk_weight_mode=args.chunk_weight_mode,
        segment_multipliers=segment_multipliers,
        round_multipliers=round_multipliers,
        current_round=args.current_round,
        current_round_multiplier=args.current_round_multiplier,
    )

    stats = report["dataset_stats"]
    weights = report["weight_stats"]
    checks = report["acceptance_checks"]
    print(f"Wrote targets: {target_path}")
    print(f"Wrote report: {report_path}")
    print(
        "Episodes: "
        f"{stats['num_episodes']} total, {stats['success_episodes']} success, "
        f"{stats['failed_episodes']} failed, handover_success_rate={stats['handover_success_rate']:.3f}"
    )
    print(f"Frame segments: {stats['segment_counts_by_frame']}")
    print(f"AWR weight stats: {weights['awr_weight']}")
    print(f"ACT chunk weight stats: {weights['act_awr_weight']}")
    print(f"Gradient mass: {weights['gradient_mass_by_round_and_segment']}")
    print(f"Acceptance checks: {checks}")


if __name__ == "__main__":
    main()
