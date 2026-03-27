#!/usr/bin/env python

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .episode import EdgeEpisodeRecord, EdgeEpisodeStepRecord


@dataclass(frozen=True)
class SpoolLayout:
    root: Path
    pending_dirname: str = "pending"
    sealed_dirname: str = "sealed"
    uploaded_dirname: str = "uploaded"

    @property
    def pending(self) -> Path:
        return self.root / self.pending_dirname

    @property
    def sealed(self) -> Path:
        return self.root / self.sealed_dirname

    @property
    def uploaded(self) -> Path:
        return self.root / self.uploaded_dirname


class EdgeEpisodeSpool:
    """Durable local spool for episode-first upload flows."""

    def __init__(self, root: str | Path):
        self.layout = SpoolLayout(root=Path(root))
        self.layout.pending.mkdir(parents=True, exist_ok=True)
        self.layout.sealed.mkdir(parents=True, exist_ok=True)
        self.layout.uploaded.mkdir(parents=True, exist_ok=True)

    def create_episode(self, episode: EdgeEpisodeRecord) -> Path:
        episode_dir = self.layout.pending / episode.episode_id
        episode_dir.mkdir(parents=True, exist_ok=False)
        self._write_json(episode_dir / "meta.json", episode.to_dict())
        self._write_json(episode_dir / "upload_state.json", {"status": "open", "last_step_idx": -1})
        (episode_dir / "steps.jsonl").touch()
        return episode_dir

    def append_step(self, step: EdgeEpisodeStepRecord) -> None:
        episode_dir = self.layout.pending / step.episode_id
        if not episode_dir.exists():
            raise FileNotFoundError(f"Pending episode '{step.episode_id}' does not exist in spool.")

        with (episode_dir / "steps.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(step.to_dict(), sort_keys=True))
            handle.write("\n")

        upload_state_path = episode_dir / "upload_state.json"
        upload_state = self._read_json(upload_state_path)
        upload_state["last_step_idx"] = step.step_idx
        self._write_json(upload_state_path, upload_state)

    def seal_episode(self, episode_id: str, summary: dict | None = None) -> Path:
        pending_dir = self.layout.pending / episode_id
        if not pending_dir.exists():
            raise FileNotFoundError(f"Pending episode '{episode_id}' does not exist in spool.")

        upload_state_path = pending_dir / "upload_state.json"
        upload_state = self._read_json(upload_state_path)
        upload_state["status"] = "sealed"
        self._write_json(upload_state_path, upload_state)
        if summary is not None:
            self._write_json(pending_dir / "summary.json", summary)

        sealed_dir = self.layout.sealed / episode_id
        pending_dir.rename(sealed_dir)
        return sealed_dir

    def mark_uploaded(self, episode_id: str, upload_receipt: dict) -> Path:
        sealed_dir = self.layout.sealed / episode_id
        if not sealed_dir.exists():
            raise FileNotFoundError(f"Sealed episode '{episode_id}' does not exist in spool.")

        self._write_json(sealed_dir / "upload_receipt.json", upload_receipt)
        upload_state_path = sealed_dir / "upload_state.json"
        upload_state = self._read_json(upload_state_path)
        upload_state["status"] = "uploaded"
        self._write_json(upload_state_path, upload_state)

        uploaded_dir = self.layout.uploaded / episode_id
        sealed_dir.rename(uploaded_dir)
        return uploaded_dir

    def list_pending_episode_ids(self) -> list[str]:
        return sorted(path.name for path in self.layout.pending.iterdir() if path.is_dir())

    def list_sealed_episode_ids(self) -> list[str]:
        return sorted(path.name for path in self.layout.sealed.iterdir() if path.is_dir())

    def _write_json(self, path: Path, payload: dict) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _read_json(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
