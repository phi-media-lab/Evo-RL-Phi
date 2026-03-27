#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from .episode import EdgeEpisodeRecord, EdgeEpisodeStepRecord
from .spool import EdgeEpisodeSpool


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class EdgeEpisodeRecorder:
    """Append-only recorder that writes episode steps into the local durable spool."""

    spool: EdgeEpisodeSpool
    episode: EdgeEpisodeRecord
    step_idx: int = 0
    _sealed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.spool.create_episode(self.episode)

    @property
    def episode_id(self) -> str:
        return self.episode.episode_id

    def append_step(
        self,
        *,
        observation: dict[str, Any],
        action: Any,
        teleop_override: bool,
        reward_raw: float = 0.0,
        done_local: bool = False,
        done_train: bool = False,
        info: dict[str, Any] | None = None,
        obs_ts: str | None = None,
        action_ts: str | None = None,
        reward_relabel: float | None = None,
        reward_relabel_version: str | None = None,
    ) -> EdgeEpisodeStepRecord:
        if self._sealed:
            raise RuntimeError("Cannot append to a sealed episode.")

        step = EdgeEpisodeStepRecord(
            episode_id=self.episode_id,
            step_idx=self.step_idx,
            obs_ts=obs_ts or _utcnow_iso(),
            action_ts=action_ts or _utcnow_iso(),
            teleop_override=teleop_override,
            reward_raw=reward_raw,
            reward_relabel=reward_relabel,
            reward_relabel_version=reward_relabel_version,
            done_local=done_local,
            done_train=done_train,
            observation=_json_safe(observation),
            action=_json_safe(action),
            info=_json_safe(info or {}),
        )
        self.spool.append_step(step)
        self.step_idx += 1
        return step

    def seal(self, summary: dict[str, Any] | None = None) -> None:
        if self._sealed:
            return
        self.spool.seal_episode(
            self.episode_id,
            summary=_json_safe(summary or {"step_count": self.step_idx}),
        )
        self._sealed = True
