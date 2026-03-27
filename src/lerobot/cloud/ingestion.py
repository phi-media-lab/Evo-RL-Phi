#!/usr/bin/env python

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import request


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class EpisodeChunk:
    """Idempotent upload unit for edge-to-cloud episode transfer."""

    episode_id: str
    chunk_index: int
    payload_digest: str
    byte_size: int
    step_count: int
    created_at: str = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id cannot be empty.")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be >= 0.")
        if self.byte_size < 0:
            raise ValueError("byte_size must be >= 0.")
        if self.step_count < 0:
            raise ValueError("step_count must be >= 0.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeCommitRequest:
    """Commit message sent after all chunks for an episode are durable in cloud storage."""

    episode_id: str
    expected_chunk_count: int
    final_step_idx: int
    created_at: str = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id cannot be empty.")
        if self.expected_chunk_count <= 0:
            raise ValueError("expected_chunk_count must be positive.")
        if self.final_step_idx < 0:
            raise ValueError("final_step_idx must be >= 0.")

    def to_dict(self) -> dict:
        return asdict(self)


class FilesystemEpisodeIngestionStore:
    """Simple local sink that emulates cloud durability for chunk + commit flows."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def upload_chunk(self, chunk: EpisodeChunk, payload: str) -> Path:
        episode_dir = self.root / chunk.episode_id / "chunks"
        episode_dir.mkdir(parents=True, exist_ok=True)

        payload_path = episode_dir / f"{chunk.chunk_index:06d}.jsonl"
        payload_path.write_text(payload, encoding="utf-8")

        meta_path = episode_dir / f"{chunk.chunk_index:06d}.meta.json"
        self._write_json(meta_path, chunk.to_dict())
        return payload_path

    def commit_episode(
        self,
        request: EpisodeCommitRequest,
        *,
        episode_meta: dict[str, Any],
        episode_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        episode_dir = self.root / request.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(episode_dir / "episode_meta.json", episode_meta)
        if episode_summary is not None:
            self._write_json(episode_dir / "episode_summary.json", episode_summary)
        self._write_json(episode_dir / "commit.json", request.to_dict())

        receipt = {
            "episode_id": request.episode_id,
            "expected_chunk_count": request.expected_chunk_count,
            "final_step_idx": request.final_step_idx,
            "status": "committed",
        }
        self._write_json(episode_dir / "receipt.json", receipt)
        return receipt

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


class HTTPEpisodeIngestionClient:
    """HTTP client that talks to the minimal ingestion service."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def upload_chunk(self, chunk: EpisodeChunk, payload: str) -> dict[str, Any]:
        return self._post_json(
            "/chunks",
            {
                "chunk": chunk.to_dict(),
                "payload": payload,
            },
        )

    def commit_episode(
        self,
        request_payload: EpisodeCommitRequest,
        *,
        episode_meta: dict[str, Any],
        episode_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._post_json(
            "/commit",
            {
                "request": request_payload.to_dict(),
                "episode_meta": episode_meta,
                "episode_summary": episode_summary,
            },
        )

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))


class EpisodeIngestionHTTPRequestHandler(BaseHTTPRequestHandler):
    """Request handler for the minimal ingestion service."""

    store: FilesystemEpisodeIngestionStore

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        payload = json.loads(raw_body.decode("utf-8"))

        if self.path == "/chunks":
            chunk = EpisodeChunk(**payload["chunk"])
            payload_path = self.store.upload_chunk(chunk, payload["payload"])
            self._write_json(
                HTTPStatus.OK,
                {
                    "episode_id": chunk.episode_id,
                    "chunk_index": chunk.chunk_index,
                    "payload_path": str(payload_path),
                },
            )
            return

        if self.path == "/commit":
            commit = EpisodeCommitRequest(**payload["request"])
            receipt = self.store.commit_episode(
                commit,
                episode_meta=payload["episode_meta"],
                episode_summary=payload.get("episode_summary"),
            )
            self._write_json(HTTPStatus.OK, receipt)
            return

        self._write_json(HTTPStatus.NOT_FOUND, {"error": f"unknown path: {self.path}"})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class EpisodeIngestionHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server that exposes FilesystemEpisodeIngestionStore over HTTP."""

    def __init__(self, server_address: tuple[str, int], store: FilesystemEpisodeIngestionStore):
        handler_cls = type(
            "BoundEpisodeIngestionHTTPRequestHandler",
            (EpisodeIngestionHTTPRequestHandler,),
            {"store": store},
        )
        super().__init__(server_address, handler_cls)
