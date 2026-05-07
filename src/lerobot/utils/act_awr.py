#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.utils.rabc import resolve_hf_path


class ACTAWRWeights:
    """Load precomputed ACT-AWR weights and return per-sample batch weights.

    The target parquet must contain:
        - ``index``: global LeRobot frame index.
        - ``weight_column``: sample weight, usually ``act_awr_weight``.

    This class does not compute advantage. It only bridges a sidecar target file
    into the existing per-sample weighted ACT training path.
    """

    def __init__(
        self,
        targets_path: str | Path,
        weight_column: str = "act_awr_weight",
        missing_weight: float = 1.0,
        normalize_batch: bool = True,
        epsilon: float = 1e-6,
        device: torch.device | None = None,
    ):
        self.targets_path = resolve_hf_path(targets_path)
        self.weight_column = weight_column
        self.missing_weight = float(missing_weight)
        self.normalize_batch = normalize_batch
        self.epsilon = float(epsilon)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info("Loading ACT-AWR weights from %s", self.targets_path)
        df = pd.read_parquet(self.targets_path)
        if "index" not in df.columns:
            raise ValueError(f"ACT-AWR target file {self.targets_path} missing 'index' column")
        if weight_column not in df.columns:
            raise ValueError(
                f"ACT-AWR target file {self.targets_path} missing weight column '{weight_column}'. "
                f"Available columns: {list(df.columns)}"
            )
        df = df[["index", weight_column]]
        if df["index"].duplicated().any():
            dupes = df.loc[df["index"].duplicated(), "index"].head(10).tolist()
            raise ValueError(f"ACT-AWR target file has duplicate indices, first duplicates: {dupes}")

        weights = df[weight_column].astype(float)
        if weights.isna().any():
            raise ValueError(f"ACT-AWR target file has NaN values in '{weight_column}'")
        if (weights < 0).any():
            raise ValueError(f"ACT-AWR target file has negative values in '{weight_column}'")

        self.weight_lookup = {
            int(index): float(weight)
            for index, weight in zip(df["index"].astype(int).to_numpy(), weights.to_numpy(), strict=True)
        }
        self.num_frames = len(self.weight_lookup)
        self.weight_min = float(weights.min())
        self.weight_mean = float(weights.mean())
        self.weight_max = float(weights.max())
        self.last_missing_count = 0

        logging.info(
            "Loaded %d ACT-AWR weights from %s: min=%.6f mean=%.6f max=%.6f",
            self.num_frames,
            self.targets_path,
            self.weight_min,
            self.weight_mean,
            self.weight_max,
        )

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        indices = batch.get("index")
        if indices is None:
            logging.warning("ACT-AWR: batch missing 'index' key, using uniform weights")
            batch_size = self._get_batch_size(batch)
            weights = torch.ones(batch_size, device=self.device, dtype=torch.float32)
            return weights, {
                "raw_mean_weight": 1.0,
                "raw_min_weight": 1.0,
                "raw_max_weight": 1.0,
                "missing_count": batch_size,
            }

        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().reshape(-1).numpy().tolist()
        elif isinstance(indices, np.ndarray):
            indices = indices.reshape(-1).tolist()
        elif isinstance(indices, (list, tuple)):
            indices = list(indices)
        else:
            indices = [indices]

        raw_weights = np.empty(len(indices), dtype=np.float32)
        missing_count = 0
        for i, idx in enumerate(indices):
            idx = int(idx)
            weight = self.weight_lookup.get(idx)
            if weight is None:
                weight = self.missing_weight
                missing_count += 1
            raw_weights[i] = weight

        self.last_missing_count = missing_count
        stats = {
            "raw_mean_weight": float(np.mean(raw_weights)) if len(raw_weights) else 1.0,
            "raw_min_weight": float(np.min(raw_weights)) if len(raw_weights) else 1.0,
            "raw_max_weight": float(np.max(raw_weights)) if len(raw_weights) else 1.0,
            "missing_count": missing_count,
        }

        weights = torch.tensor(raw_weights, device=self.device, dtype=torch.float32)
        if self.normalize_batch:
            batch_size = len(weights)
            weights = weights * batch_size / (weights.sum() + self.epsilon)
        return weights, stats

    def _get_batch_size(self, batch: dict) -> int:
        for key in ["action", "index"]:
            if key in batch:
                value = batch[key]
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    return int(value.shape[0])
        return 1

    def get_stats(self) -> dict:
        return {
            "num_frames": self.num_frames,
            "targets_path": str(self.targets_path),
            "weight_column": self.weight_column,
            "weight_min": self.weight_min,
            "weight_mean": self.weight_mean,
            "weight_max": self.weight_max,
            "missing_weight": self.missing_weight,
            "normalize_batch": self.normalize_batch,
            "last_missing_count": self.last_missing_count,
        }
