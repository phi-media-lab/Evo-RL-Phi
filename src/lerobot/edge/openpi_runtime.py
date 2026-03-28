#!/usr/bin/env python

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .openpi_loader import resolve_openpi_hybrid_bridge_feed_builder, resolve_openpi_mlx_hybrid_bridge_feed_builder
from .runtime_protocol import EdgePolicyRuntime, EdgeRuntimeFactory


def _default_openpi_client_root() -> Path:
    return Path("/Users/fbsh/ane-openpi/openpi/packages/openpi-client/src")


def _default_openpi_repo_root() -> Path:
    return Path("/Users/fbsh/ane-openpi/openpi")


@dataclass(frozen=True)
class OpenPIObservationAdapterConfig:
    mode: str = "flat_state"
    state_keys: list[str] = field(default_factory=list)
    state_key: str = ""
    state_start_index: int = 0
    state_dim: int | None = None
    image_keys: list[str] = field(default_factory=list)
    prompt: str = ""
    prompt_key: str = ""
    image_aliases: dict[str, str] = field(default_factory=dict)
    drop_batch_dim: bool = True
    transpose_images_to_chw: bool = False

    def __post_init__(self) -> None:
        if self.mode == "flat_state" and not self.state_keys:
            raise ValueError("state_keys cannot be empty for flat_state mode.")
        if self.mode in {"aloha_raw", "openpi_raw"} and not self.state_key:
            raise ValueError(f"state_key cannot be empty for {self.mode} mode.")
        if self.mode == "openpi_raw" and not self.image_keys:
            raise ValueError("image_keys cannot be empty for openpi_raw mode.")
        if self.mode not in {"flat_state", "aloha_raw", "openpi_raw"}:
            raise ValueError(f"Unsupported OpenPI observation adapter mode: {self.mode}")


class OpenPIObservationAdapter:
    def __init__(self, config: OpenPIObservationAdapterConfig):
        self.config = config

    def adapt(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.config.mode == "flat_state":
            adapted = self._adapt_flat_state(observation)
        elif self.config.mode == "aloha_raw":
            adapted = self._adapt_aloha_raw(observation)
        else:
            adapted = self._adapt_openpi_raw(observation)
        self._copy_token_fields(observation, adapted)
        prompt = self._resolve_prompt(observation)
        if prompt:
            adapted["prompt"] = prompt
        return adapted

    def _adapt_flat_state(self, observation: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray([observation[key] for key in self.config.state_keys], dtype=np.float32)
        adapted: dict[str, Any] = {"state": state}
        for key in self.config.image_keys:
            if key not in observation:
                raise KeyError(f"Missing image observation key: {key}")
            image_name = self.config.image_aliases.get(key, key)
            adapted[image_name] = np.asarray(observation[key], dtype=np.uint8)
        return adapted

    def _adapt_aloha_raw(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.config.state_key not in observation:
            raise KeyError(f"Missing state observation key: {self.config.state_key}")
        state = self._extract_array(observation[self.config.state_key], dtype=np.float32)
        if state.ndim != 1:
            raise ValueError(f"Expected 1D state after batch squeeze, got shape {tuple(state.shape)}")
        start = self.config.state_start_index
        end = None if self.config.state_dim is None else start + self.config.state_dim
        state = state[start:end]
        images: dict[str, np.ndarray] = {}
        for key in self.config.image_keys:
            if key not in observation:
                raise KeyError(f"Missing image observation key: {key}")
            image_name = self.config.image_aliases.get(key, key)
            images[image_name] = self._adapt_image(observation[key])
        return {
            "state": state,
            "images": images,
        }

    def _adapt_openpi_raw(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.config.state_key not in observation:
            raise KeyError(f"Missing state observation key: {self.config.state_key}")
        state = np.asarray(observation[self.config.state_key], dtype=np.float32)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        elif state.ndim != 2:
            raise ValueError(f"Expected 1D or 2D raw state, got shape {tuple(state.shape)}")
        batch_size = state.shape[0]
        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.ndarray] = {}
        for key in self.config.image_keys:
            if key not in observation:
                raise KeyError(f"Missing image observation key: {key}")
            image_name = self.config.image_aliases.get(key, key.removeprefix("image."))
            image = np.asarray(observation[key], dtype=np.uint8)
            if image.ndim == 3:
                image = np.expand_dims(image, axis=0)
            elif image.ndim != 4:
                raise ValueError(f"Expected 3D or 4D raw image, got shape {tuple(image.shape)}")
            if image.shape[0] != batch_size:
                raise ValueError(
                    f"Image batch dimension mismatch for '{key}': expected {batch_size}, got {image.shape[0]}"
                )
            images[image_name] = image
            image_masks[image_name] = np.ones((batch_size,), dtype=bool)
        return {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

    def _resolve_prompt(self, observation: dict[str, Any]) -> str:
        if self.config.prompt_key:
            value = observation.get(self.config.prompt_key)
            if value is not None:
                if isinstance(value, str):
                    return value
                value_array = self._extract_array(value)
                if np.isscalar(value_array):
                    return str(value_array.item())
                return str(value_array)
        return self.config.prompt

    def _extract_array(self, value: Any, *, dtype: Any | None = None) -> np.ndarray:
        array = np.asarray(value, dtype=dtype)
        if self.config.drop_batch_dim and array.ndim > 0 and array.shape[0] == 1:
            array = array[0]
        return array

    def _adapt_image(self, value: Any) -> np.ndarray:
        image = self._extract_array(value, dtype=np.uint8)
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image after batch squeeze, got shape {tuple(image.shape)}")
        if self.config.transpose_images_to_chw:
            if image.shape[-1] in {1, 3, 4}:
                return np.transpose(image, (2, 0, 1))
            if image.shape[0] in {1, 3, 4}:
                return image
            raise ValueError(f"Cannot infer image channel axis for shape {tuple(image.shape)}")
        return image

    def _copy_token_fields(self, observation: dict[str, Any], adapted: dict[str, Any]) -> None:
        for key in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
            value = observation.get(key)
            if value is not None:
                adapted[key] = self._extract_array(value)


@dataclass(frozen=True)
class OpenPIWebsocketRuntimeConfig:
    server_uri: str
    action_horizon: int
    action_dim: int
    observation: OpenPIObservationAdapterConfig
    openpi_client_root: str = field(default_factory=lambda: str(_default_openpi_client_root()))

    def __post_init__(self) -> None:
        if not self.server_uri:
            raise ValueError("server_uri cannot be empty.")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive.")


@dataclass(frozen=True)
class OpenPIInProcessRuntimeConfig:
    action_horizon: int
    action_dim: int
    observation: OpenPIObservationAdapterConfig
    policy_ref: str = ""

    def __post_init__(self) -> None:
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if not self.policy_ref:
            raise ValueError("policy_ref cannot be empty.")


@dataclass(frozen=True)
class OpenPICoreMLRuntimeConfig:
    model_path: str
    action_horizon: int
    action_dim: int
    observation: OpenPIObservationAdapterConfig
    model_output_key: str = ""
    output_transform: str = ""
    compute_unit: str = "cpu_and_ne"
    openpi_repo_root: str = field(default_factory=lambda: str(_default_openpi_repo_root()))

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("model_path cannot be empty.")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive.")


@dataclass(frozen=True)
class OpenPIHybridBridgeRuntimeConfig:
    model_path: str
    bridge_mode: str
    action_horizon: int
    action_dim: int
    observation: OpenPIObservationAdapterConfig
    model_output_key: str = ""
    output_transform: str = ""
    feed_contract_path: str = ""
    prefix_mode: str = ""
    compute_unit: str = "cpu_and_ne"
    openpi_repo_root: str = field(default_factory=lambda: str(_default_openpi_repo_root()))
    config_name: str = ""
    checkpoint_dir: str = ""
    export_precision: str = "float16"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("model_path cannot be empty.")
        if not self.bridge_mode:
            raise ValueError("bridge_mode cannot be empty.")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive.")


def _compute_coreml_compute_unit(name: str) -> Any:
    import coremltools as ct

    return {
        "all": ct.ComputeUnit.ALL,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[name]


def _load_compiled_mlmodel_predictor(model_path: str, *, compute_unit: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    import coremltools as ct

    init_timings: dict[str, float] = {}
    model_path_obj = Path(model_path).resolve()
    compiled_path = model_path_obj.with_suffix(".mlmodelc")
    if not compiled_path.exists():
        started = time.perf_counter()
        compiled_path = Path(ct.models.utils.compile_model(str(model_path_obj), str(compiled_path))).resolve()
        init_timings["compile_model_s"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    compiled_model = ct.models.CompiledMLModel(str(compiled_path), compute_units=_compute_coreml_compute_unit(compute_unit))
    init_timings["load_compiled_model_s"] = round(time.perf_counter() - started, 6)
    started = time.perf_counter()
    source_model = ct.models.MLModel(str(model_path_obj), skip_model_load=True)
    init_timings["load_source_model_s"] = round(time.perf_counter() - started, 6)
    input_specs = list(source_model.get_spec().description.input)
    input_names = [spec.name for spec in input_specs]
    input_shapes = {spec.name: tuple(spec.type.multiArrayType.shape) for spec in input_specs}
    output_specs = list(source_model.get_spec().description.output)
    output_names = [spec.name for spec in output_specs]

    class CompiledMLModelPredictor:
        def __init__(self) -> None:
            self.initialization_timings = dict(init_timings)
            self.last_timings: dict[str, float] | None = None

        def __call__(self, feed: dict[str, Any]) -> dict[str, Any]:
            started = time.perf_counter()
            missing = [name for name in input_names if name not in feed]
            if missing:
                raise ValueError(f"CoreML feed is missing required inputs: {missing}")
            runtime_feed = {name: feed[name] for name in input_names}
            for name, value in runtime_feed.items():
                expected_shape = input_shapes.get(name)
                actual_shape = tuple(np.asarray(value).shape)
                if expected_shape and actual_shape != expected_shape:
                    raise ValueError(
                        f"CoreML input shape mismatch for '{name}': expected {expected_shape}, got {actual_shape}"
                    )
            validate_s = round(time.perf_counter() - started, 6)

            started = time.perf_counter()
            prediction = compiled_model.predict(runtime_feed)
            predict_s = round(time.perf_counter() - started, 6)
            unexpected_outputs = sorted(set(prediction) - set(output_names))
            if unexpected_outputs:
                raise ValueError(
                    f"CoreML predictor returned unexpected outputs {unexpected_outputs}; expected subset of {output_names}"
                )
            self.last_timings = {
                "validate_feed_s": validate_s,
                "predict_s": predict_s,
            }
            return prediction

    return CompiledMLModelPredictor()


def _collect_callable_timings(obj: Any, *, prefix: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    initialization_timings = getattr(obj, "initialization_timings", None)
    if isinstance(initialization_timings, dict):
        for key, value in initialization_timings.items():
            timings[f"{prefix}_init_{key}"] = float(value)
    last_timings = getattr(obj, "last_timings", None)
    if isinstance(last_timings, dict):
        for key, value in last_timings.items():
            timings[f"{prefix}_{key}"] = float(value)
    return timings


def _normalize_aloha(value: torch.Tensor, *, min_val: float, max_val: float) -> torch.Tensor:
    return (value - min_val) / (max_val - min_val)


def _unnormalize_aloha(value: torch.Tensor, *, min_val: float, max_val: float) -> torch.Tensor:
    return value * (max_val - min_val) + min_val


def _aloha_gripper_from_angular(value: torch.Tensor) -> torch.Tensor:
    value = value + 0.5476
    return _normalize_aloha(value, min_val=-0.6213, max_val=1.4910)


def _encode_aloha_actions_from_openpi_internal(actions: torch.Tensor) -> torch.Tensor:
    joint_flip_mask = torch.tensor(
        [1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1],
        dtype=actions.dtype,
        device=actions.device,
    )
    encoded = joint_flip_mask * actions[..., :14]
    encoded[..., 6] = _aloha_gripper_from_angular(encoded[..., 6])
    encoded[..., 13] = _aloha_gripper_from_angular(encoded[..., 13])
    return encoded


def _extract_openpi_response_array(
    response: dict[str, Any] | np.ndarray,
    *,
    model_output_key: str = "",
) -> Any:
    if not isinstance(response, dict):
        return response
    if "actions" in response:
        return response["actions"]
    if model_output_key:
        if model_output_key not in response:
            raise KeyError(f"OpenPI runtime response missing configured output key '{model_output_key}'.")
        return response[model_output_key]
    if len(response) == 1:
        return next(iter(response.values()))
    raise KeyError("OpenPI runtime response missing 'actions' and no model_output_key was configured.")


def _apply_openpi_output_transform(chunk: torch.Tensor, *, output_transform: str) -> torch.Tensor:
    if not output_transform:
        return chunk
    if output_transform == "aloha_actions":
        if chunk.shape[-1] < 14:
            raise ValueError(f"Aloha output transform requires at least 14 dims, got {chunk.shape[-1]}")
        return _encode_aloha_actions_from_openpi_internal(chunk)
    raise ValueError(f"Unsupported OpenPI output transform: {output_transform}")


def _load_openpi_client_policy_class(openpi_client_root: str) -> Any:
    client_root = Path(openpi_client_root)
    if not client_root.exists():
        raise FileNotFoundError(f"openpi client root does not exist: {client_root}")
    client_root_str = str(client_root)
    if client_root_str not in sys.path:
        sys.path.insert(0, client_root_str)
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    return WebsocketClientPolicy


class OpenPIWebsocketRuntime:
    def __init__(
        self,
        config: OpenPIWebsocketRuntimeConfig,
        *,
        client_factory: Callable[[OpenPIWebsocketRuntimeConfig], Any] | None = None,
    ):
        self.config = config
        self.adapter = OpenPIObservationAdapter(config.observation)
        self.action_queue: list[torch.Tensor] = []
        self._client_factory = client_factory or self._build_default_client
        self.client = self._client_factory(config)

    @property
    def queue_size(self) -> int:
        return len(self.action_queue)

    def reset(self) -> None:
        self.action_queue.clear()
        reset_fn = getattr(self.client, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def warmup(self, observation: dict[str, Any] | None = None) -> torch.Tensor:
        return self.refill_action_queue(observation or {})

    def maybe_refill_action_queue(self, observation: dict[str, Any] | None = None) -> torch.Tensor | None:
        if self.action_queue:
            return None
        return self.refill_action_queue(observation or {})

    def refill_action_queue(self, observation: dict[str, Any] | None = None) -> torch.Tensor:
        adapted_observation = self.adapter.adapt(observation or {})
        response = self.client.infer(adapted_observation)
        chunk = _coerce_action_chunk(
            response,
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
        )
        self.action_queue.extend(list(chunk.unbind(0)))
        return chunk

    def pop_next_action(self, observation: dict[str, Any] | None = None) -> torch.Tensor:
        self.maybe_refill_action_queue(observation or {})
        if not self.action_queue:
            self.refill_action_queue(observation or {})
        return self.action_queue.pop(0)

    def _build_default_client(self, config: OpenPIWebsocketRuntimeConfig) -> Any:
        WebsocketClientPolicy = _load_openpi_client_policy_class(config.openpi_client_root)
        if config.server_uri.startswith("ws://"):
            host = config.server_uri.removeprefix("ws://")
        elif config.server_uri.startswith("wss://"):
            host = config.server_uri
        else:
            host = config.server_uri
        return WebsocketClientPolicy(host=host)


class OpenPIInProcessRuntime:
    def __init__(
        self,
        config: OpenPIInProcessRuntimeConfig,
        *,
        policy: Any,
    ):
        self.config = config
        self.policy = policy
        self.adapter = OpenPIObservationAdapter(config.observation)
        self.action_queue: list[torch.Tensor] = []

    @property
    def queue_size(self) -> int:
        return len(self.action_queue)

    def reset(self) -> None:
        self.action_queue.clear()
        reset_fn = getattr(self.policy, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def warmup(self, observation: dict[str, Any] | None = None) -> torch.Tensor:
        return self.refill_action_queue(observation or {})

    def maybe_refill_action_queue(self, observation: dict[str, Any] | None = None) -> torch.Tensor | None:
        if self.action_queue:
            return None
        return self.refill_action_queue(observation or {})

    def refill_action_queue(self, observation: dict[str, Any] | None = None) -> torch.Tensor:
        adapted_observation = self.adapter.adapt(observation or {})
        response = self.policy.infer(adapted_observation)
        chunk = _coerce_action_chunk(
            response,
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
        )
        self.action_queue.extend(list(chunk.unbind(0)))
        return chunk

    def pop_next_action(self, observation: dict[str, Any] | None = None) -> torch.Tensor:
        self.maybe_refill_action_queue(observation or {})
        if not self.action_queue:
            self.refill_action_queue(observation or {})
        return self.action_queue.pop(0)


class OpenPIHybridBridgeRuntime:
    def __init__(
        self,
        config: OpenPIHybridBridgeRuntimeConfig,
        *,
        predictor: Callable[[dict[str, Any]], dict[str, Any] | np.ndarray],
        feed_builder: Callable[[dict[str, Any], OpenPIHybridBridgeRuntimeConfig], dict[str, Any]] | None = None,
    ):
        self.config = config
        self.predictor = predictor
        self.feed_builder = feed_builder or self._default_feed_builder
        self.adapter = OpenPIObservationAdapter(config.observation)
        self.action_queue: list[torch.Tensor] = []
        self.last_runtime_info: dict[str, float] = {}

    @property
    def queue_size(self) -> int:
        return len(self.action_queue)

    def reset(self) -> None:
        self.action_queue.clear()
        self.last_runtime_info = {}

    def warmup(self, observation: dict[str, Any] | None = None) -> torch.Tensor:
        return self.refill_action_queue(observation or {})

    def maybe_refill_action_queue(self, observation: dict[str, Any] | None = None) -> torch.Tensor | None:
        if self.action_queue:
            return None
        return self.refill_action_queue(observation or {})

    def refill_action_queue(self, observation: dict[str, Any] | None = None) -> torch.Tensor:
        self.last_runtime_info = {}
        adapted_observation = self.adapter.adapt(observation or {})
        started = time.perf_counter()
        feed = self.feed_builder(adapted_observation, self.config)
        self.last_runtime_info["feed_builder_total_s"] = round(time.perf_counter() - started, 6)
        self.last_runtime_info.update(_collect_callable_timings(self.feed_builder, prefix="feed_builder"))
        started = time.perf_counter()
        response = self.predictor(feed)
        self.last_runtime_info["predictor_total_s"] = round(time.perf_counter() - started, 6)
        self.last_runtime_info.update(_collect_callable_timings(self.predictor, prefix="predictor"))
        chunk = _coerce_action_chunk(
            response,
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            model_output_key=self.config.model_output_key,
            output_transform=self.config.output_transform,
        )
        self.action_queue.extend(list(chunk.unbind(0)))
        return chunk

    def pop_next_action(self, observation: dict[str, Any] | None = None) -> torch.Tensor:
        self.maybe_refill_action_queue(observation or {})
        if not self.action_queue:
            self.refill_action_queue(observation or {})
        return self.action_queue.pop(0)

    def _default_feed_builder(
        self,
        adapted_observation: dict[str, Any],
        config: OpenPIHybridBridgeRuntimeConfig,
    ) -> dict[str, Any]:
        return {
            "observation": adapted_observation,
            "model_path": config.model_path,
            "feed_contract_path": config.feed_contract_path,
            "bridge_mode": config.bridge_mode,
            "prefix_mode": config.prefix_mode,
            "compute_unit": config.compute_unit,
        }


def _coerce_action_chunk(
    response: dict[str, Any] | np.ndarray,
    *,
    action_dim: int,
    action_horizon: int,
    model_output_key: str = "",
    output_transform: str = "",
) -> torch.Tensor:
    actions = _extract_openpi_response_array(response, model_output_key=model_output_key)
    chunk = torch.as_tensor(actions, dtype=torch.float32)
    if chunk.ndim == 3:
        if chunk.shape[0] != 1:
            raise ValueError(f"Expected batch dimension of size 1, got shape {tuple(chunk.shape)}")
        chunk = chunk[0]
    if chunk.ndim == 1:
        chunk = chunk.unsqueeze(0)
    if chunk.ndim != 2:
        raise ValueError(f"Expected 2D action chunk, got shape {tuple(chunk.shape)}")
    chunk = _apply_openpi_output_transform(chunk, output_transform=output_transform)
    if chunk.shape[1] != action_dim:
        raise ValueError(f"Action dimension mismatch: expected {action_dim}, got {chunk.shape[1]}")
    return chunk[:action_horizon]


class OpenPICoreMLRuntime:
    def __init__(
        self,
        config: OpenPICoreMLRuntimeConfig,
        *,
        predictor: Callable[[dict[str, Any]], dict[str, Any] | np.ndarray],
        feed_builder: Callable[[dict[str, Any], OpenPICoreMLRuntimeConfig], dict[str, Any]] | None = None,
    ):
        self.config = config
        self.predictor = predictor
        self.feed_builder = feed_builder or self._default_feed_builder
        self.adapter = OpenPIObservationAdapter(config.observation)
        self.action_queue: list[torch.Tensor] = []
        self.last_runtime_info: dict[str, float] = {}

    @property
    def queue_size(self) -> int:
        return len(self.action_queue)

    def reset(self) -> None:
        self.action_queue.clear()
        self.last_runtime_info = {}

    def warmup(self, observation: dict[str, Any] | None = None) -> torch.Tensor:
        return self.refill_action_queue(observation or {})

    def maybe_refill_action_queue(self, observation: dict[str, Any] | None = None) -> torch.Tensor | None:
        if self.action_queue:
            return None
        return self.refill_action_queue(observation or {})

    def refill_action_queue(self, observation: dict[str, Any] | None = None) -> torch.Tensor:
        self.last_runtime_info = {}
        adapted_observation = self.adapter.adapt(observation or {})
        started = time.perf_counter()
        feed = self.feed_builder(adapted_observation, self.config)
        self.last_runtime_info["feed_builder_total_s"] = round(time.perf_counter() - started, 6)
        self.last_runtime_info.update(_collect_callable_timings(self.feed_builder, prefix="feed_builder"))
        started = time.perf_counter()
        response = self.predictor(feed)
        self.last_runtime_info["predictor_total_s"] = round(time.perf_counter() - started, 6)
        self.last_runtime_info.update(_collect_callable_timings(self.predictor, prefix="predictor"))
        chunk = _coerce_action_chunk(
            response,
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            model_output_key=self.config.model_output_key,
            output_transform=self.config.output_transform,
        )
        self.action_queue.extend(list(chunk.unbind(0)))
        return chunk

    def pop_next_action(self, observation: dict[str, Any] | None = None) -> torch.Tensor:
        self.maybe_refill_action_queue(observation or {})
        if not self.action_queue:
            self.refill_action_queue(observation or {})
        return self.action_queue.pop(0)

    def _default_feed_builder(
        self,
        adapted_observation: dict[str, Any],
        config: OpenPICoreMLRuntimeConfig,
    ) -> dict[str, Any]:
        return {
            "observation": adapted_observation,
            "model_path": config.model_path,
            "compute_unit": config.compute_unit,
        }


class OpenPIWebsocketRuntimeFactory:
    runtime_config_type = OpenPIWebsocketRuntimeConfig

    def build_runtime(
        self,
        runtime_config: OpenPIWebsocketRuntimeConfig,
        *,
        warmup_observation: dict[str, Any] | None = None,
    ) -> EdgePolicyRuntime:
        runtime = OpenPIWebsocketRuntime(runtime_config)
        if warmup_observation is not None:
            runtime.warmup(warmup_observation)
        return runtime

    def supports_runtime(self, runtime: EdgePolicyRuntime) -> bool:
        return isinstance(runtime, OpenPIWebsocketRuntime)


class OpenPIInProcessRuntimeFactory:
    runtime_config_type = OpenPIInProcessRuntimeConfig

    def __init__(self, policy_resolver: Callable[[str], Any]):
        self.policy_resolver = policy_resolver

    def build_runtime(
        self,
        runtime_config: OpenPIInProcessRuntimeConfig,
        *,
        warmup_observation: dict[str, Any] | None = None,
    ) -> EdgePolicyRuntime:
        policy = self.policy_resolver(runtime_config.policy_ref)
        runtime = OpenPIInProcessRuntime(runtime_config, policy=policy)
        if warmup_observation is not None:
            runtime.warmup(warmup_observation)
        return runtime

    def supports_runtime(self, runtime: EdgePolicyRuntime) -> bool:
        return isinstance(runtime, OpenPIInProcessRuntime)


class OpenPIHybridBridgeRuntimeFactory:
    runtime_config_type = OpenPIHybridBridgeRuntimeConfig

    def __init__(
        self,
        predictor_factory: Callable[[OpenPIHybridBridgeRuntimeConfig], Callable[[dict[str, Any]], dict[str, Any] | np.ndarray]]
        | None = None,
        feed_builder_factory: Callable[
            [OpenPIHybridBridgeRuntimeConfig],
            Callable[[dict[str, Any], OpenPIHybridBridgeRuntimeConfig], dict[str, Any]],
        ]
        | None = None,
        feed_builder: Callable[[dict[str, Any], OpenPIHybridBridgeRuntimeConfig], dict[str, Any]] | None = None,
    ):
        self.predictor_factory = predictor_factory or self._default_predictor_factory
        self.feed_builder_factory = feed_builder_factory or self._default_feed_builder_factory
        self.feed_builder = feed_builder

    def build_runtime(
        self,
        runtime_config: OpenPIHybridBridgeRuntimeConfig,
        *,
        warmup_observation: dict[str, Any] | None = None,
    ) -> EdgePolicyRuntime:
        predictor = self.predictor_factory(runtime_config)
        feed_builder = self.feed_builder or self.feed_builder_factory(runtime_config)
        runtime = OpenPIHybridBridgeRuntime(runtime_config, predictor=predictor, feed_builder=feed_builder)
        if warmup_observation is not None:
            runtime.warmup(warmup_observation)
        return runtime

    def supports_runtime(self, runtime: EdgePolicyRuntime) -> bool:
        return isinstance(runtime, OpenPIHybridBridgeRuntime)

    def _default_predictor_factory(
        self,
        runtime_config: OpenPIHybridBridgeRuntimeConfig,
    ) -> Callable[[dict[str, Any]], dict[str, Any] | np.ndarray]:
        return _load_compiled_mlmodel_predictor(
            runtime_config.model_path,
            compute_unit=runtime_config.compute_unit,
        )

    def _default_feed_builder_factory(
        self,
        runtime_config: OpenPIHybridBridgeRuntimeConfig,
    ) -> Callable[[dict[str, Any], OpenPIHybridBridgeRuntimeConfig], dict[str, Any]]:
        if runtime_config.config_name and runtime_config.checkpoint_dir:
            if runtime_config.bridge_mode == "mlx_full_prefix_hostbridge":
                return resolve_openpi_mlx_hybrid_bridge_feed_builder(
                    openpi_repo_root=runtime_config.openpi_repo_root,
                    config_name=runtime_config.config_name,
                    checkpoint_dir=runtime_config.checkpoint_dir,
                    device=runtime_config.device,
                    export_precision=runtime_config.export_precision,
                )
            return resolve_openpi_hybrid_bridge_feed_builder(
                openpi_repo_root=runtime_config.openpi_repo_root,
                config_name=runtime_config.config_name,
                checkpoint_dir=runtime_config.checkpoint_dir,
                device=runtime_config.device,
                prefix_mode=runtime_config.prefix_mode,
                export_precision=runtime_config.export_precision,
            )
        return lambda adapted_observation, config: {
            "observation": adapted_observation,
            "model_path": config.model_path,
            "feed_contract_path": config.feed_contract_path,
            "bridge_mode": config.bridge_mode,
            "prefix_mode": config.prefix_mode,
            "compute_unit": config.compute_unit,
        }


class OpenPICoreMLRuntimeFactory:
    runtime_config_type = OpenPICoreMLRuntimeConfig

    def __init__(
        self,
        predictor_factory: Callable[[OpenPICoreMLRuntimeConfig], Callable[[dict[str, Any]], dict[str, Any] | np.ndarray]]
        | None = None,
        feed_builder: Callable[[dict[str, Any], OpenPICoreMLRuntimeConfig], dict[str, Any]] | None = None,
    ):
        self.predictor_factory = predictor_factory or self._default_predictor_factory
        self.feed_builder = feed_builder

    def build_runtime(
        self,
        runtime_config: OpenPICoreMLRuntimeConfig,
        *,
        warmup_observation: dict[str, Any] | None = None,
    ) -> EdgePolicyRuntime:
        predictor = self.predictor_factory(runtime_config)
        runtime = OpenPICoreMLRuntime(runtime_config, predictor=predictor, feed_builder=self.feed_builder)
        if warmup_observation is not None:
            runtime.warmup(warmup_observation)
        return runtime

    def supports_runtime(self, runtime: EdgePolicyRuntime) -> bool:
        return isinstance(runtime, OpenPICoreMLRuntime)

    def _default_predictor_factory(
        self,
        runtime_config: OpenPICoreMLRuntimeConfig,
    ) -> Callable[[dict[str, Any]], dict[str, Any] | np.ndarray]:
        return _load_compiled_mlmodel_predictor(
            runtime_config.model_path,
            compute_unit=runtime_config.compute_unit,
        )


class CompositeEdgeRuntimeFactory:
    def __init__(self, *factories: EdgeRuntimeFactory):
        self.factories = list(factories)

    def build_runtime(
        self,
        runtime_config: Any,
        *,
        warmup_observation: dict[str, Any] | None = None,
    ) -> EdgePolicyRuntime:
        for factory in self.factories:
            runtime_config_type = getattr(factory, "runtime_config_type", None)
            if runtime_config_type is not None and isinstance(runtime_config, runtime_config_type):
                return factory.build_runtime(runtime_config, warmup_observation=warmup_observation)
        raise TypeError(f"Unsupported runtime config type: {type(runtime_config).__name__}")

    def supports_runtime(self, runtime: EdgePolicyRuntime) -> bool:
        return any(factory.supports_runtime(runtime) for factory in self.factories)
