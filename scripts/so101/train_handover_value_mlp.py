#!/usr/bin/env python

"""Train a lightweight scalar SO101 handover value model from value_targets_v0.parquet."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class FeatureStats:
    mean: list[float]
    std: list[float]
    feature_names: list[str]


class ValueMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FrameValueDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
    ):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "target": self.targets[idx],
            "weight": self.weights[idx],
        }


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


def _state_to_array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        # Pandas can roundtrip list columns as strings in some parquet engines.
        value = json.loads(value)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _build_features(df: pd.DataFrame, *, include_action: bool) -> tuple[np.ndarray, list[str]]:
    states = np.stack([_state_to_array(value) for value in df["observation_state"].tolist()]).astype(np.float32)
    feature_blocks = [states]
    feature_names = [f"state_{i}" for i in range(states.shape[1])]

    progress = df["frame_progress"].to_numpy(dtype=np.float32).reshape(-1, 1)
    feature_blocks.append(progress)
    feature_names.append("frame_progress")

    # Smooth low-frequency time basis. This gives v0 enough temporal context without leaking labels.
    feature_blocks.append(np.sin(progress * math.pi).astype(np.float32))
    feature_blocks.append(np.cos(progress * math.pi).astype(np.float32))
    feature_names.extend(["sin_progress_pi", "cos_progress_pi"])

    if include_action and "action" in df.columns:
        actions = np.stack([_state_to_array(value) for value in df["action"].tolist()]).astype(np.float32)
        feature_blocks.append(actions)
        feature_names.extend([f"action_{i}" for i in range(actions.shape[1])])

    return np.concatenate(feature_blocks, axis=1), feature_names


def _load_visual_embeddings(path: Path, *, expected_rows: int) -> tuple[np.ndarray, list[str]]:
    data = np.load(path)
    if "embeddings" not in data:
        raise KeyError(f"Visual embeddings file {path} is missing key 'embeddings'.")
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Visual embeddings must be rank-2, got shape={embeddings.shape}.")
    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Visual embeddings row count mismatch: expected {expected_rows}, got {embeddings.shape[0]}."
        )
    names = [f"visual_embedding_{i}" for i in range(embeddings.shape[1])]
    return embeddings, names


def _episode_split(df: pd.DataFrame, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    episodes = sorted(df["source_episode_uid"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * val_ratio))) if len(episodes) > 1 else 0
    val_episodes = set(episodes[:val_count])
    is_val = df["source_episode_uid"].isin(val_episodes).to_numpy()
    return ~is_val, is_val


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (weight * (pred - target).pow(2)).sum() / weight.sum().clamp_min(1e-6)


def _compute_metrics(
    df: pd.DataFrame,
    pred: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    *,
    prefix: str,
) -> dict[str, Any]:
    err = pred - target
    weighted_mse = float(np.sum(weight * err**2) / max(float(np.sum(weight)), 1e-8))
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    if len(pred) > 1 and np.std(pred) > 1e-8 and np.std(target) > 1e-8:
        corr = float(np.corrcoef(pred, target)[0, 1])
    else:
        corr = 0.0

    out: dict[str, Any] = {
        f"{prefix}_rows": int(len(df)),
        f"{prefix}_episodes": int(df["source_episode_uid"].nunique()),
        f"{prefix}_mse": mse,
        f"{prefix}_weighted_mse": weighted_mse,
        f"{prefix}_mae": mae,
        f"{prefix}_corr": corr,
    }

    by_segment = []
    tmp = df[["segment_role", "source_episode_uid", "handover_success", "failure_type"]].copy()
    tmp["_pred"] = pred
    tmp["_target"] = target
    tmp["_abs_err"] = np.abs(err)
    for segment, group in tmp.groupby("segment_role"):
        by_segment.append(
            {
                "segment_role": segment,
                "rows": int(len(group)),
                "episodes": int(group["source_episode_uid"].nunique()),
                "target_mean": float(group["_target"].mean()),
                "pred_mean": float(group["_pred"].mean()),
                "mae": float(group["_abs_err"].mean()),
            }
        )
    out[f"{prefix}_by_segment"] = sorted(by_segment, key=lambda row: row["segment_role"])
    return out


@torch.no_grad()
def _predict(model: nn.Module, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(features), batch_size):
        x = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
        preds.append(model(x).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-column", default="value_target_segment")
    parser.add_argument("--weight-column", default="value_loss_weight")
    parser.add_argument("--include-action", action="store_true")
    parser.add_argument("--visual-embeddings", type=Path, default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    df = pd.read_parquet(args.targets)
    for column in ["observation_state", args.target_column, args.weight_column, "source_episode_uid"]:
        if column not in df.columns:
            raise KeyError(f"Targets parquet is missing required column: {column}")
    df = df[df[args.weight_column].astype(float) > 0].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows with positive value loss weight.")

    features, feature_names = _build_features(df, include_action=args.include_action)
    if args.visual_embeddings is not None:
        visual_features, visual_names = _load_visual_embeddings(args.visual_embeddings, expected_rows=len(df))
        features = np.concatenate([features, visual_features], axis=1)
        feature_names.extend(visual_names)
    targets = df[args.target_column].to_numpy(dtype=np.float32)
    weights = df[args.weight_column].to_numpy(dtype=np.float32)

    train_mask, val_mask = _episode_split(df, val_ratio=args.val_ratio, seed=args.seed)
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Train/val split produced an empty split.")

    mean = features[train_mask].mean(axis=0)
    std = features[train_mask].std(axis=0)
    std = np.maximum(std, 1e-6)
    features_norm = (features - mean) / std

    train_ds = FrameValueDataset(features_norm[train_mask], targets[train_mask], weights[train_mask])
    val_ds = FrameValueDataset(features_norm[val_mask], targets[val_mask], weights[val_mask])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = ValueMLP(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    losses = []
    loader_iter = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        model.train()
        x = batch["features"].to(device)
        y = batch["target"].to(device)
        w = batch["weight"].to(device)
        pred = model(x)
        loss = _weighted_mse(pred, y, w)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

        if step == 1 or step % max(1, args.steps // 10) == 0:
            print(f"step={step} train_weighted_mse={losses[-1]:.6f}")

    pred_all = _predict(model, features_norm, batch_size=args.batch_size, device=device)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pt"
    metrics_path = out_dir / "metrics.json"
    config_path = out_dir / "config.json"
    predictions_path = out_dir / "value_predictions.parquet"

    metrics = {
        "schema_version": "so101_handover_value_mlp.v0",
        "targets": str(args.targets),
        "target_column": args.target_column,
        "weight_column": args.weight_column,
        "train": _compute_metrics(
            df.loc[train_mask].reset_index(drop=True),
            pred_all[train_mask],
            targets[train_mask],
            weights[train_mask],
            prefix="train",
        ),
        "val": _compute_metrics(
            df.loc[val_mask].reset_index(drop=True),
            pred_all[val_mask],
            targets[val_mask],
            weights[val_mask],
            prefix="val",
        ),
        "loss_last": float(losses[-1]),
        "loss_mean_last_100": float(np.mean(losses[-100:])),
        "device": str(device),
    }

    config = {
        "input_dim": int(features.shape[1]),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "include_action": args.include_action,
        "visual_embeddings": str(args.visual_embeddings) if args.visual_embeddings else None,
        "feature_stats": asdict(FeatureStats(mean=mean.tolist(), std=std.tolist(), feature_names=feature_names)),
        "target_column": args.target_column,
        "weight_column": args.weight_column,
    }
    torch.save({"model_state_dict": model.state_dict(), "config": config, "metrics": metrics}, model_path)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    pred_df = df[
        [
            "source_name",
            "episode_index",
            "episode_id",
            "source_episode_uid",
            "global_index",
            "frame_index",
            "episode_length",
            "frame_progress",
            "handover_success",
            "failure_type",
            "segment_role",
            "collection_round",
            "source_policy_generation",
            args.target_column,
            args.weight_column,
        ]
    ].copy()
    pred_df["split"] = np.where(val_mask, "val", "train")
    pred_df["value_pred"] = pred_all.astype(np.float32)
    pred_df["value_error"] = (pred_df["value_pred"].to_numpy(dtype=np.float32) - targets).astype(np.float32)
    pred_df.to_parquet(predictions_path, index=False)

    print(f"Wrote model to {model_path}")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote predictions to {predictions_path}")
    print(
        "val: "
        f"mse={metrics['val']['val_mse']:.6f} "
        f"mae={metrics['val']['val_mae']:.6f} "
        f"corr={metrics['val']['val_corr']:.4f}"
    )


if __name__ == "__main__":
    main()
