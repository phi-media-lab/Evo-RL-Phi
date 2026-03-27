#!/usr/bin/env python

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MaterializedEpisodeSummary:
    episode_id: str
    step_count: int
    policy_artifact_id: str
    task_id: str
    robot_id: str
    source_commit_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaterializedDatasetManifest:
    episode_count: int
    total_step_count: int
    episodes: list[MaterializedEpisodeSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_count": self.episode_count,
            "total_step_count": self.total_step_count,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }


class FilesystemEpisodeMaterializer:
    """Materializes committed ingestion episodes into a deterministic dataset directory."""

    def __init__(self, ingestion_root: str | Path, output_root: str | Path):
        self.ingestion_root = Path(ingestion_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def materialize_all(self) -> MaterializedDatasetManifest:
        episodes: list[MaterializedEpisodeSummary] = []
        total_step_count = 0

        for episode_dir in sorted(path for path in self.ingestion_root.iterdir() if path.is_dir()):
            summary = self.materialize_episode(episode_dir.name)
            episodes.append(summary)
            total_step_count += summary.step_count

        manifest = MaterializedDatasetManifest(
            episode_count=len(episodes),
            total_step_count=total_step_count,
            episodes=episodes,
        )
        self._write_json(self.output_root / "manifest.json", manifest.to_dict())
        return manifest

    def materialize_episode(self, episode_id: str) -> MaterializedEpisodeSummary:
        source_episode_dir = self.ingestion_root / episode_id
        if not source_episode_dir.exists():
            raise FileNotFoundError(f"Ingestion episode '{episode_id}' does not exist.")

        commit_path = source_episode_dir / "commit.json"
        meta_path = source_episode_dir / "episode_meta.json"
        if not commit_path.exists():
            raise FileNotFoundError(f"Episode '{episode_id}' is missing commit.json.")
        if not meta_path.exists():
            raise FileNotFoundError(f"Episode '{episode_id}' is missing episode_meta.json.")

        commit_payload = self._read_json(commit_path)
        meta_payload = self._read_json(meta_path)

        materialized_episode_dir = self.output_root / "episodes" / episode_id
        materialized_episode_dir.mkdir(parents=True, exist_ok=True)

        steps = self._read_ordered_steps(source_episode_dir / "chunks")
        steps_path = materialized_episode_dir / "steps.jsonl"
        steps_path.write_text("".join(f"{json.dumps(step, sort_keys=True)}\n" for step in steps), encoding="utf-8")

        self._write_json(materialized_episode_dir / "episode_meta.json", meta_payload)
        summary = MaterializedEpisodeSummary(
            episode_id=episode_id,
            step_count=int(commit_payload["final_step_idx"]) + 1,
            policy_artifact_id=str(meta_payload["policy_artifact_id"]),
            task_id=str(meta_payload["task_id"]),
            robot_id=str(meta_payload["robot_id"]),
            source_commit_path=str(commit_path),
        )
        self._write_json(materialized_episode_dir / "summary.json", summary.to_dict())
        return summary

    def _read_ordered_steps(self, chunks_dir: Path) -> list[dict[str, Any]]:
        if not chunks_dir.exists():
            raise FileNotFoundError(f"Missing chunks directory: {chunks_dir}")

        steps: list[dict[str, Any]] = []
        for chunk_path in sorted(chunks_dir.glob("*.jsonl")):
            with chunk_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    steps.append(json.loads(line))
        return steps

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _read_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
