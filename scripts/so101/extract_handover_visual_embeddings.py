#!/usr/bin/env python

"""Extract frozen dual-camera visual embeddings for SO101 handover value targets.

The extractor uses HIL manifests to map each target row to the correct episode video segment. It
supports shared video files by using the per-episode `from_timestamp_s` offset in the manifest.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


CAMERA_KEYS = ["observation.images.front_cam", "observation.images.hand_cam"]


@dataclass(frozen=True)
class ManifestSpec:
    name: str
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


def _parse_manifest_spec(raw: str) -> ManifestSpec:
    parts: dict[str, str] = {}
    for token in raw.split("::"):
        if "=" not in token:
            raise ValueError(f"Invalid --source-manifest '{raw}'. Expected name=...::manifest=...")
        key, value = token.split("=", maxsplit=1)
        parts[key.strip()] = value.strip()
    if "name" not in parts or "manifest" not in parts:
        raise ValueError(f"Invalid --source-manifest '{raw}'. Expected name=...::manifest=...")
    return ManifestSpec(name=parts["name"], manifest=Path(parts["manifest"]).expanduser())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _load_manifest_map(specs: list[ManifestSpec]) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for spec in specs:
        rows = _load_jsonl(spec.manifest)
        for row in rows:
            out[(spec.name, int(row["episode_index"]))] = row
    return out


def _parse_path_rewrites(raw_values: list[str]) -> list[tuple[str, str]]:
    rewrites: list[tuple[str, str]] = []
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"Invalid --path-rewrite '{raw}'. Expected OLD=NEW.")
        old, new = raw.split("=", maxsplit=1)
        rewrites.append((old, new))
    # Longest prefixes first.
    return sorted(rewrites, key=lambda item: len(item[0]), reverse=True)


def _rewrite_path(path: str, rewrites: list[tuple[str, str]]) -> Path:
    rewritten = path
    for old, new in rewrites:
        if rewritten.startswith(old):
            rewritten = new + rewritten[len(old) :]
            break
    return Path(rewritten)


def _make_encoder(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        dim = int(model.fc.in_features)
    elif name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        dim = int(model.fc.in_features)
    else:
        raise ValueError(f"Unsupported encoder '{name}'. Use resnet18 or resnet50.")
    model.fc = nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, dim


def _preprocess_frames(frames_bgr: list[np.ndarray], image_size: int, device: torch.device) -> torch.Tensor:
    arrs = []
    for frame in frames_bgr:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        arrs.append(resized)
    batch = np.stack(arrs).astype(np.float32) / 255.0
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


@torch.no_grad()
def _encode_batch(
    encoder: nn.Module,
    frames_bgr: list[np.ndarray],
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    batch = _preprocess_frames(frames_bgr, image_size=image_size, device=device)
    features = encoder(batch).detach().cpu().numpy().astype(np.float32)
    return features


def _video_fps(cap: cv2.VideoCapture, fallback: float) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-3 or math.isnan(fps):
        return fallback
    return fps


def _extract_for_video(
    *,
    video_path: Path,
    requests: list[tuple[int, int]],
    encoder: nn.Module,
    image_size: int,
    batch_size: int,
    device: torch.device,
    feature_dim: int,
    features_out: np.ndarray,
    camera_offset: int,
) -> dict[str, Any]:
    if not video_path.exists():
        return {
            "video_path": str(video_path),
            "requested": len(requests),
            "filled": 0,
            "missing": len(requests),
            "error": "missing_video",
        }

    by_frame: dict[int, list[int]] = {}
    for video_frame_index, row_index in requests:
        by_frame.setdefault(int(video_frame_index), []).append(int(row_index))
    needed = set(by_frame)
    if not needed:
        return {"video_path": str(video_path), "requested": 0, "filled": 0, "missing": 0}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video_path": str(video_path),
            "requested": len(requests),
            "filled": 0,
            "missing": len(requests),
            "error": "open_failed",
        }

    max_needed = max(needed)
    frames: list[np.ndarray] = []
    row_indices: list[int] = []
    filled = 0
    current = 0
    while current <= max_needed:
        ok, frame = cap.read()
        if not ok:
            break
        if current in needed:
            for row_index in by_frame[current]:
                frames.append(frame.copy())
                row_indices.append(row_index)
            if len(frames) >= batch_size:
                encoded = _encode_batch(encoder, frames, image_size=image_size, device=device)
                for i, row_index in enumerate(row_indices):
                    start = camera_offset
                    end = camera_offset + feature_dim
                    features_out[row_index, start:end] = encoded[i].astype(np.float16)
                filled += len(row_indices)
                frames.clear()
                row_indices.clear()
        current += 1

    if frames:
        encoded = _encode_batch(encoder, frames, image_size=image_size, device=device)
        for i, row_index in enumerate(row_indices):
            start = camera_offset
            end = camera_offset + feature_dim
            features_out[row_index, start:end] = encoded[i].astype(np.float16)
        filled += len(row_indices)

    cap.release()
    return {
        "video_path": str(video_path),
        "requested": len(requests),
        "filled": int(filled),
        "missing": int(len(requests) - filled),
        "max_requested_frame": int(max_needed),
        "fps": _video_fps(cap, fallback=30.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", action="append", required=True)
    parser.add_argument("--path-rewrite", action="append", default=[])
    parser.add_argument("--encoder", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--limit-rows", type=int, default=None)
    args = parser.parse_args()

    targets = pd.read_parquet(args.targets).reset_index(drop=True)
    if args.limit_rows is not None:
        targets = targets.iloc[: int(args.limit_rows)].reset_index(drop=True)
    for column in ["source_name", "episode_index", "frame_index", "timestamp"]:
        if column not in targets.columns:
            raise KeyError(f"Targets parquet is missing required column: {column}")

    manifest_specs = [_parse_manifest_spec(raw) for raw in args.source_manifest]
    manifest_map = _load_manifest_map(manifest_specs)
    rewrites = _parse_path_rewrites(args.path_rewrite)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    encoder, feature_dim = _make_encoder(args.encoder, pretrained=not args.no_pretrained)
    encoder = encoder.to(device)

    features = np.full((len(targets), len(CAMERA_KEYS) * feature_dim), np.nan, dtype=np.float16)
    camera_reports: list[dict[str, Any]] = []
    missing_episode_rows = 0

    for camera_i, camera_key in enumerate(CAMERA_KEYS):
        requests_by_video: dict[Path, list[tuple[int, int]]] = {}
        for row_index, row in targets.iterrows():
            source_name = str(row["source_name"])
            episode_index = int(row["episode_index"])
            manifest_row = manifest_map.get((source_name, episode_index))
            if manifest_row is None:
                missing_episode_rows += 1
                continue
            videos = manifest_row.get("videos") or {}
            video_info = videos.get(camera_key)
            if not isinstance(video_info, dict) or not video_info.get("path"):
                missing_episode_rows += 1
                continue
            video_path = _rewrite_path(str(video_info["path"]), rewrites)
            from_ts = float(video_info.get("from_timestamp_s") or 0.0)
            timestamp = float(row["timestamp"])
            video_frame_index = int(round((from_ts + timestamp) * args.fps))
            requests_by_video.setdefault(video_path, []).append((video_frame_index, int(row_index)))

        for video_num, (video_path, requests) in enumerate(sorted(requests_by_video.items())):
            report = _extract_for_video(
                video_path=video_path,
                requests=requests,
                encoder=encoder,
                image_size=args.image_size,
                batch_size=args.batch_size,
                device=device,
                feature_dim=feature_dim,
                features_out=features,
                camera_offset=camera_i * feature_dim,
            )
            report["camera_key"] = camera_key
            report["video_num"] = video_num
            camera_reports.append(report)
            if (video_num + 1) % 10 == 0 or video_num == 0:
                print(
                    f"{camera_key} video {video_num + 1}/{len(requests_by_video)} "
                    f"filled={report.get('filled')} missing={report.get('missing')} path={video_path.name}"
                )

    filled_by_camera = {}
    for camera_i, camera_key in enumerate(CAMERA_KEYS):
        start = camera_i * feature_dim
        end = start + feature_dim
        mask = np.isfinite(features[:, start:end]).all(axis=1)
        filled_by_camera[camera_key] = int(mask.sum())

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = out_dir / f"visual_embeddings_{args.encoder}_fp16.npz"
    meta_path = out_dir / "visual_embeddings_meta.parquet"
    report_path = out_dir / "visual_embeddings_report.json"

    np.savez_compressed(
        embeddings_path,
        embeddings=features,
        camera_keys=np.asarray(CAMERA_KEYS, dtype=object),
        encoder=np.asarray([args.encoder], dtype=object),
        feature_dim=np.asarray([feature_dim], dtype=np.int64),
    )
    targets[
        [
            "source_name",
            "episode_index",
            "episode_id",
            "source_episode_uid",
            "global_index",
            "frame_index",
            "timestamp",
            "handover_success",
            "failure_type",
            "segment_role",
        ]
    ].to_parquet(meta_path, index=False)
    report = {
        "schema_version": "so101_handover_visual_embeddings.v0",
        "targets": str(args.targets),
        "embeddings_path": str(embeddings_path),
        "meta_path": str(meta_path),
        "rows": int(len(targets)),
        "encoder": args.encoder,
        "pretrained": not args.no_pretrained,
        "image_size": args.image_size,
        "feature_dim_per_camera": feature_dim,
        "camera_keys": CAMERA_KEYS,
        "filled_by_camera": filled_by_camera,
        "missing_episode_rows": int(missing_episode_rows),
        "path_rewrites": rewrites,
        "camera_reports": camera_reports,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(f"Wrote embeddings to {embeddings_path}")
    print(f"Wrote report to {report_path}")
    print(f"filled_by_camera={filled_by_camera}")


if __name__ == "__main__":
    main()
