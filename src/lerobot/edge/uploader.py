#!/usr/bin/env python

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.cloud.ingestion import EpisodeChunk, EpisodeCommitRequest, FilesystemEpisodeIngestionStore
from lerobot.control_plane.artifact import compute_sha256, compute_sha256_bytes

from .spool import EdgeEpisodeSpool


@dataclass(frozen=True)
class EdgeUploaderConfig:
    """Chunking policy for uploading sealed episodes."""

    steps_per_chunk: int = 64

    def __post_init__(self) -> None:
        if self.steps_per_chunk <= 0:
            raise ValueError("steps_per_chunk must be positive.")


class EdgeEpisodeUploader:
    """Uploads sealed episodes from the local spool into a durable sink."""

    def __init__(
        self,
        *,
        spool: EdgeEpisodeSpool,
        sink: FilesystemEpisodeIngestionStore,
        config: EdgeUploaderConfig | None = None,
    ):
        self.spool = spool
        self.sink = sink
        self.config = config or EdgeUploaderConfig()

    def upload_episode(self, episode_id: str) -> dict[str, Any]:
        episode_dir = self.spool.layout.sealed / episode_id
        if not episode_dir.exists():
            raise FileNotFoundError(f"Sealed episode '{episode_id}' does not exist in spool.")

        meta = self._read_json(episode_dir / "meta.json")
        summary = self._read_optional_json(episode_dir / "summary.json")
        steps = (episode_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
        if not steps:
            raise ValueError(f"Sealed episode '{episode_id}' contains no steps.")

        chunks = self._build_chunks(episode_id=episode_id, step_lines=steps)
        for chunk, payload in chunks:
            self.sink.upload_chunk(chunk, payload)

        commit = EpisodeCommitRequest(
            episode_id=episode_id,
            expected_chunk_count=len(chunks),
            final_step_idx=len(steps) - 1,
        )
        receipt = self.sink.commit_episode(
            commit,
            episode_meta=meta,
            episode_summary=summary,
        )
        receipt["chunks"] = [chunk.to_dict() for chunk, _ in chunks]
        receipt["meta_digest"] = compute_sha256(episode_dir / "meta.json")
        receipt["steps_digest"] = compute_sha256(episode_dir / "steps.jsonl")
        self.spool.mark_uploaded(episode_id, upload_receipt=receipt)
        return receipt

    def upload_all_sealed(self) -> list[dict[str, Any]]:
        return [self.upload_episode(episode_id) for episode_id in self.spool.list_sealed_episode_ids()]

    def _build_chunks(self, *, episode_id: str, step_lines: list[str]) -> list[tuple[EpisodeChunk, str]]:
        chunks: list[tuple[EpisodeChunk, str]] = []
        for chunk_index, chunk_start in enumerate(range(0, len(step_lines), self.config.steps_per_chunk)):
            lines = step_lines[chunk_start : chunk_start + self.config.steps_per_chunk]
            payload = "\n".join(lines) + "\n"
            payload_bytes = payload.encode("utf-8")

            chunk = EpisodeChunk(
                episode_id=episode_id,
                chunk_index=chunk_index,
                payload_digest=compute_sha256_bytes(payload_bytes),
                byte_size=len(payload_bytes),
                step_count=len(lines),
            )
            chunks.append((chunk, payload))
        return chunks

    def _read_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _read_optional_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return self._read_json(path)
