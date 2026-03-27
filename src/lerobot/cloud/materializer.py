#!/usr/bin/env python

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import request as urllib_request

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE, REWARD


def _sanitize_feature_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def _to_numpy(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.float64:
        return array.astype(np.float32)
    if array.dtype == np.int64:
        return array.astype(np.int32)
    return array


def _flatten_numeric_dict(prefix: str, payload: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    flattened: list[tuple[str, np.ndarray]] = []
    for key in sorted(payload):
        value = payload[key]
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.extend(_flatten_numeric_dict(child_prefix, value))
            continue
        array = _to_numpy(value)
        if array.ndim == 3 and array.shape[-1] in {1, 3, 4} and array.dtype == np.uint8:
            flattened.append((child_prefix, array))
            continue
        flattened.append((child_prefix, array.reshape(-1).astype(np.float32)))
    return flattened


def _infer_lerobot_features(
    step: dict[str, Any],
    *,
    include_env_state_alias: bool,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    observation = step.get("observation") or {}
    action = step.get("action")

    if not isinstance(observation, dict):
        raise ValueError("Expected step observation to be a dictionary.")

    flattened_observation = _flatten_numeric_dict("", observation)
    image_features = {
        f"{OBS_IMAGES}.{_sanitize_feature_name(name)}": {
            "dtype": "image",
            "shape": tuple(array.shape),
            "names": ["height", "width", "channel"],
        }
        for name, array in flattened_observation
        if array.ndim == 3
    }
    state_entries = [(name, array) for name, array in flattened_observation if array.ndim != 3]
    if not state_entries:
        raise ValueError("Materializer requires at least one numeric observation entry.")

    state_names: list[str] = []
    state_parts: list[np.ndarray] = []
    for name, array in state_entries:
        dim = int(array.size)
        if dim == 1:
            state_names.append(name)
        else:
            state_names.extend(f"{name}[{idx}]" for idx in range(dim))
        state_parts.append(array.astype(np.float32))

    if isinstance(action, dict):
        flattened_action = _flatten_numeric_dict("", action)
        action_names: list[str] = []
        action_parts: list[np.ndarray] = []
        for name, array in flattened_action:
            if array.ndim == 3:
                raise ValueError("Action images are not supported in materialization.")
            dim = int(array.size)
            if dim == 1:
                action_names.append(name)
            else:
                action_names.extend(f"{name}[{idx}]" for idx in range(dim))
            action_parts.append(array.astype(np.float32))
        if not action_parts:
            raise ValueError("Materializer requires at least one numeric action entry.")
        action_dim = int(sum(part.size for part in action_parts))
    else:
        action_array = _to_numpy(action)
        action_array = action_array.reshape(-1).astype(np.float32)
        action_dim = int(action_array.size)
        action_names = [f"action[{idx}]" for idx in range(action_dim)]
        if action_dim == 0:
            raise ValueError("Materializer requires non-empty action payloads.")

    features: dict[str, dict[str, Any]] = {
        ACTION: {"dtype": "float32", "shape": (action_dim,), "names": action_names},
        OBS_STATE: {
            "dtype": "float32",
            "shape": (int(sum(part.size for part in state_parts)),),
            "names": state_names,
        },
        REWARD: {"dtype": "float32", "shape": (1,), "names": None},
        DONE: {"dtype": "bool", "shape": (1,), "names": None},
        "complementary_info.done_local": {"dtype": "bool", "shape": (1,), "names": None},
        "complementary_info.teleop_override": {"dtype": "bool", "shape": (1,), "names": None},
    }
    if include_env_state_alias:
        features[OBS_ENV_STATE] = {
            "dtype": "float32",
            "shape": (int(sum(part.size for part in state_parts)),),
            "names": list(state_names),
        }
    features.update(image_features)
    return features, state_names, action_names


def _step_to_lerobot_frame(
    step: dict[str, Any],
    *,
    state_names: list[str],
    action_names: list[str],
    task: str,
    include_env_state_alias: bool,
) -> dict[str, Any]:
    observation = step.get("observation") or {}
    action = step.get("action")

    flattened_observation = _flatten_numeric_dict("", observation)
    state_values: list[np.ndarray] = []
    frame: dict[str, Any] = {"task": task}

    for name, array in flattened_observation:
        if array.ndim == 3:
            frame[f"{OBS_IMAGES}.{_sanitize_feature_name(name)}"] = array
            continue
        state_values.append(array.astype(np.float32))

    flattened_state_names: list[str] = []
    for name, array in [(name, array) for name, array in flattened_observation if array.ndim != 3]:
        if array.size == 1:
            flattened_state_names.append(name)
        else:
            flattened_state_names.extend(f"{name}[{idx}]" for idx in range(int(array.size)))
    if flattened_state_names != state_names:
        raise ValueError("Observation schema drift detected while materializing episodes.")

    state_vector = np.concatenate(state_values).astype(np.float32)
    frame[OBS_STATE] = state_vector
    if include_env_state_alias:
        frame[OBS_ENV_STATE] = state_vector.copy()

    if isinstance(action, dict):
        action_parts = _flatten_numeric_dict("", action)
        flattened_action_names: list[str] = []
        action_values: list[np.ndarray] = []
        for name, array in action_parts:
            if array.ndim == 3:
                raise ValueError("Action images are not supported in materialization.")
            if array.size == 1:
                flattened_action_names.append(name)
            else:
                flattened_action_names.extend(f"{name}[{idx}]" for idx in range(int(array.size)))
            action_values.append(array.astype(np.float32))
        if flattened_action_names != action_names:
            raise ValueError("Action schema drift detected while materializing episodes.")
        frame[ACTION] = np.concatenate(action_values).astype(np.float32)
    else:
        action_array = _to_numpy(action).reshape(-1).astype(np.float32)
        flattened_action_names = [f"action[{idx}]" for idx in range(int(action_array.size))]
        if flattened_action_names != action_names:
            raise ValueError("Action schema drift detected while materializing episodes.")
        frame[ACTION] = action_array

    frame[REWARD] = np.asarray([float(step.get("reward_raw", 0.0))], dtype=np.float32)
    frame[DONE] = np.asarray([bool(step.get("done_train", False))], dtype=bool)
    frame["complementary_info.done_local"] = np.asarray([bool(step.get("done_local", False))], dtype=bool)
    frame["complementary_info.teleop_override"] = np.asarray(
        [bool(step.get("teleop_override", False))], dtype=bool
    )
    return frame


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
    lerobot_repo_id: str | None = None
    lerobot_root: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_count": self.episode_count,
            "total_step_count": self.total_step_count,
            "lerobot_repo_id": self.lerobot_repo_id,
            "lerobot_root": self.lerobot_root,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }


class FilesystemEpisodeMaterializer:
    """Materializes committed ingestion episodes into deterministic local dataset layouts."""

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

    def materialize_to_lerobot_dataset(
        self,
        *,
        repo_id: str,
        dataset_root: str | Path,
        fps: int,
        use_videos: bool = False,
        include_env_state_alias: bool = True,
    ) -> MaterializedDatasetManifest:
        episode_dirs = sorted(path for path in self.ingestion_root.iterdir() if path.is_dir())
        if not episode_dirs:
            raise ValueError("No committed episodes found under ingestion_root.")

        first_steps = self._read_ordered_steps(episode_dirs[0] / "chunks")
        if not first_steps:
            raise ValueError(f"Episode '{episode_dirs[0].name}' has no steps to materialize.")

        features, state_names, action_names = _infer_lerobot_features(
            first_steps[0],
            include_env_state_alias=include_env_state_alias,
        )
        dataset_root = Path(dataset_root)
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=dataset_root,
            use_videos=use_videos,
        )

        summaries: list[MaterializedEpisodeSummary] = []
        total_step_count = 0

        for episode_dir in episode_dirs:
            episode_id = episode_dir.name
            meta_payload = self._read_json(episode_dir / "episode_meta.json")
            steps = self._read_ordered_steps(episode_dir / "chunks")
            for step in steps:
                dataset.add_frame(
                    _step_to_lerobot_frame(
                        step,
                        state_names=state_names,
                        action_names=action_names,
                        task=str(meta_payload["task_id"]),
                        include_env_state_alias=include_env_state_alias,
                    )
                )
            dataset.save_episode(
                extra_episode_metadata={
                    "edge_episode_id": episode_id,
                    "policy_artifact_id": str(meta_payload["policy_artifact_id"]),
                    "robot_id": str(meta_payload["robot_id"]),
                    "channel": meta_payload.get("channel"),
                }
            )
            summary = self.materialize_episode(episode_id)
            summaries.append(summary)
            total_step_count += summary.step_count

        dataset.finalize()

        manifest = MaterializedDatasetManifest(
            episode_count=len(summaries),
            total_step_count=total_step_count,
            episodes=summaries,
            lerobot_repo_id=repo_id,
            lerobot_root=str(dataset_root),
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


class HTTPMaterializerClient:
    """Thin HTTP client for triggering remote materialization jobs."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def materialize(
        self,
        *,
        repo_id: str = "local/edge-materialized",
        fps: int = 20,
        export_lerobot_dataset: bool = False,
        use_videos: bool = False,
        include_env_state_alias: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "repo_id": repo_id,
            "fps": fps,
            "export_lerobot_dataset": export_lerobot_dataset,
            "use_videos": use_videos,
            "include_env_state_alias": include_env_state_alias,
        }
        response = urllib_request.urlopen(
            urllib_request.Request(
                url=f"{self.base_url}/materialize",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        )
        with response:
            return json.loads(response.read().decode("utf-8"))


class MaterializerHTTPRequestHandler(BaseHTTPRequestHandler):
    server: "MaterializerHTTPServer"

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/materialize":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": f"Unsupported path: {self.path}"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length).decode("utf-8") or "{}")
        manifest = self.server.materialize(payload)
        self._write_json(
            HTTPStatus.OK,
            {
                "status": "materialized",
                "manifest": manifest.to_dict(),
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MaterializerHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server that exposes FilesystemEpisodeMaterializer over HTTP."""

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        ingestion_root: str | Path,
        output_root: str | Path,
        dataset_root: str | Path | None = None,
    ):
        self.ingestion_root = Path(ingestion_root)
        self.output_root = Path(output_root)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        super().__init__(server_address, MaterializerHTTPRequestHandler)

    def materialize(self, payload: dict[str, Any]) -> MaterializedDatasetManifest:
        materializer = FilesystemEpisodeMaterializer(
            ingestion_root=self.ingestion_root,
            output_root=self.output_root,
        )
        export_lerobot_dataset = bool(payload.get("export_lerobot_dataset", False))
        if export_lerobot_dataset:
            if self.dataset_root is None:
                raise ValueError("Server was started without dataset_root but export_lerobot_dataset was requested.")
            return materializer.materialize_to_lerobot_dataset(
                repo_id=str(payload.get("repo_id", "local/edge-materialized")),
                dataset_root=self.dataset_root,
                fps=int(payload.get("fps", 20)),
                use_videos=bool(payload.get("use_videos", False)),
                include_env_state_alias=bool(payload.get("include_env_state_alias", True)),
            )
        return materializer.materialize_all()
