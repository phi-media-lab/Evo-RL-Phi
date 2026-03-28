#!/usr/bin/env python

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time
from types import ModuleType
from contextlib import contextmanager
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch


_OPENPI_HYBRID_BRIDGE_FEED_BUILDER_CACHE: dict[
    tuple[str, str, str, str, str, str],
    Callable[[dict[str, object], object], dict[str, np.ndarray]],
] = {}
_OPENPI_HYBRID_BRIDGE_MODEL_CONTEXT_CACHE: dict[
    tuple[str, str, str, str, str, str],
    tuple[ModuleType, ModuleType, ModuleType, object, torch.device, object],
] = {}
_OPENPI_MLX_HYBRID_BRIDGE_FEED_BUILDER_CACHE: dict[
    tuple[str, str, str, str, str],
    Callable[[dict[str, object], object], dict[str, np.ndarray]],
] = {}
_OPENPI_MLX_HYBRID_BRIDGE_MODEL_CONTEXT_CACHE: dict[
    tuple[str, str, str, str, str],
    tuple[ModuleType, ModuleType, ModuleType, object, torch.device, object],
] = {}


@dataclass
class TimedFeedBuilder:
    build_fn: Callable[[dict[str, object], object], dict[str, np.ndarray]]
    initialization_timings: dict[str, float]
    last_timings: dict[str, float] | None = None

    def __call__(self, adapted_observation: dict[str, object], runtime_config: object) -> dict[str, np.ndarray]:
        started = time.perf_counter()
        feed = self.build_fn(adapted_observation, runtime_config)
        timings = dict(self.last_timings or {})
        timings["build_feed_total_s"] = round(time.perf_counter() - started, 6)
        self.last_timings = timings
        return feed


def _default_openpi_root() -> Path:
    return Path("/Users/fbsh/ane-openpi/openpi/src")


def _default_openpi_client_root() -> Path:
    return Path("/Users/fbsh/ane-openpi/openpi/packages/openpi-client/src")


def _current_repo_src() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class OpenPICheckpointPolicyRef:
    config: str
    checkpoint_dir: str
    default_prompt: str | None = None
    pytorch_device: str | None = None
    openpi_root: str = str(_default_openpi_root())
    openpi_client_root: str = str(_default_openpi_client_root())


def parse_openpi_policy_ref(policy_ref: str) -> OpenPICheckpointPolicyRef:
    parsed = urlparse(policy_ref)
    if parsed.scheme != "openpi":
        raise ValueError(f"Unsupported policy ref scheme: {parsed.scheme}")
    if parsed.netloc != "checkpoint":
        raise ValueError(f"Unsupported openpi policy ref target: {parsed.netloc}")

    params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
    config = params.get("config")
    checkpoint_dir = params.get("dir")
    if not config:
        raise ValueError("OpenPI checkpoint policy_ref missing 'config' query parameter.")
    if not checkpoint_dir:
        raise ValueError("OpenPI checkpoint policy_ref missing 'dir' query parameter.")

    return OpenPICheckpointPolicyRef(
        config=config,
        checkpoint_dir=checkpoint_dir,
        default_prompt=params.get("default_prompt"),
        pytorch_device=params.get("pytorch_device"),
        openpi_root=params.get("openpi_root", str(_default_openpi_root())),
        openpi_client_root=params.get("openpi_client_root", str(_default_openpi_client_root())),
    )


@contextmanager
def _isolated_openpi_import_context():
    repo_src = str(_current_repo_src())
    original_sys_path = list(sys.path)
    stripped_sys_path = [entry for entry in original_sys_path if Path(entry).resolve() != Path(repo_src).resolve()]
    original_lerobot_modules = {
        name: module for name, module in sys.modules.items() if name == "lerobot" or name.startswith("lerobot.")
    }
    for name in list(original_lerobot_modules):
        sys.modules.pop(name, None)
    sys.path[:] = stripped_sys_path
    try:
        yield
    finally:
        sys.path[:] = original_sys_path
        current_lerobot_modules = [name for name in sys.modules if name == "lerobot" or name.startswith("lerobot.")]
        for name in current_lerobot_modules:
            sys.modules.pop(name, None)
        sys.modules.update(original_lerobot_modules)


def _import_openpi_modules(spec: OpenPICheckpointPolicyRef) -> tuple[ModuleType, ModuleType]:
    openpi_root = Path(spec.openpi_root)
    openpi_client_root = Path(spec.openpi_client_root)
    if not openpi_root.exists():
        raise FileNotFoundError(f"openpi root does not exist: {openpi_root}")
    if not openpi_client_root.exists():
        raise FileNotFoundError(f"openpi client root does not exist: {openpi_client_root}")

    openpi_root_str = str(openpi_root)
    openpi_client_root_str = str(openpi_client_root)
    with _isolated_openpi_import_context():
        if openpi_root_str not in sys.path:
            sys.path.insert(0, openpi_root_str)
        if openpi_client_root_str not in sys.path:
            sys.path.insert(0, openpi_client_root_str)

        from openpi.policies import policy_config as policy_config_module
        from openpi.training import config as training_config_module

    return policy_config_module, training_config_module


def _import_openpi_hybrid_bridge_modules(openpi_repo_root: str) -> tuple[ModuleType, ModuleType, ModuleType]:
    repo_root = Path(openpi_repo_root)
    openpi_src = repo_root / "src"
    script_path = repo_root / "scripts" / "build_rollout_feed_artifact.py"
    if not openpi_src.exists():
        raise FileNotFoundError(f"openpi src root does not exist: {openpi_src}")
    if not script_path.exists():
        raise FileNotFoundError(f"openpi hybrid bridge script does not exist: {script_path}")

    with _isolated_openpi_import_context():
        openpi_src_str = str(openpi_src)
        if openpi_src_str not in sys.path:
            sys.path.insert(0, openpi_src_str)

        from openpi.models import model as model_module
        from openpi.shared import observation_io as observation_io_module

        spec = importlib.util.spec_from_file_location("evorl_openpi_run_hybrid_bridge_v1", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load hybrid bridge helper script from {script_path}")
        run_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_module)

    return model_module, observation_io_module, run_module


def _import_openpi_mlx_hybrid_bridge_modules(openpi_repo_root: str) -> tuple[ModuleType, ModuleType, ModuleType]:
    repo_root = Path(openpi_repo_root)
    openpi_src = repo_root / "src"
    script_path = repo_root / "scripts" / "run_hybrid_bridge_v1_mlx_prefix.py"
    if not openpi_src.exists():
        raise FileNotFoundError(f"openpi src root does not exist: {openpi_src}")
    if not script_path.exists():
        raise FileNotFoundError(f"openpi MLX hybrid bridge script does not exist: {script_path}")

    with _isolated_openpi_import_context():
        openpi_src_str = str(openpi_src)
        if openpi_src_str not in sys.path:
            sys.path.insert(0, openpi_src_str)

        from openpi.models import model as model_module
        from openpi.shared import observation_io as observation_io_module

        spec = importlib.util.spec_from_file_location("evorl_openpi_run_hybrid_bridge_v1_mlx_prefix", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load MLX hybrid bridge helper script from {script_path}")
        run_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_module)

    return model_module, observation_io_module, run_module


def _normalize_openpi_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {tuple(image.shape)}")
    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))
    elif image.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"Cannot infer image channel axis for shape {tuple(image.shape)}")
    return image


def _batchify_optional_array(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        return np.expand_dims(array, axis=0)
    if array.shape[0] == 1:
        return array
    return np.expand_dims(array, axis=0)


def _adapted_observation_to_openpi_payload(adapted_observation: dict[str, object]) -> dict[str, object]:
    if "image" in adapted_observation:
        state = np.asarray(adapted_observation["state"], dtype=np.float32)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        elif state.ndim != 2:
            raise ValueError(f"Expected 1D or 2D state array, got shape {tuple(state.shape)}")

        images = adapted_observation["image"]
        if not isinstance(images, dict) or not images:
            raise ValueError("OpenPI hybrid bridge feed requires a non-empty 'image' dict.")

        payload: dict[str, object] = {
            "state": state,
            "image": {},
            "image_mask": {},
        }
        image_masks = adapted_observation.get("image_mask", {})
        for key, value in images.items():
            image = np.asarray(value, dtype=np.uint8)
            if image.ndim == 3:
                image = np.expand_dims(image, axis=0)
            elif image.ndim != 4:
                raise ValueError(f"Expected 3D or 4D image array, got shape {tuple(image.shape)}")
            payload["image"][key] = image
            if isinstance(image_masks, dict) and key in image_masks:
                payload["image_mask"][key] = np.asarray(image_masks[key], dtype=bool)
            else:
                payload["image_mask"][key] = np.ones((image.shape[0],), dtype=bool)

        for key in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
            value = adapted_observation.get(key)
            if value is not None:
                payload[key] = _batchify_optional_array(value)
        return payload

    state = np.asarray(adapted_observation["state"], dtype=np.float32)
    if state.ndim == 1:
        state = np.expand_dims(state, axis=0)
    elif state.ndim != 2:
        raise ValueError(f"Expected 1D or 2D state array, got shape {tuple(state.shape)}")

    images = adapted_observation.get("images")
    if not isinstance(images, dict) or not images:
        raise ValueError("OpenPI hybrid bridge feed requires a non-empty 'images' dict.")

    payload: dict[str, object] = {
        "state": state,
        "image": {},
        "image_mask": {},
    }
    for key, value in images.items():
        image = _normalize_openpi_image(np.asarray(value, dtype=np.uint8))
        payload["image"][key] = np.expand_dims(image, axis=0)
        payload["image_mask"][key] = np.asarray([True], dtype=bool)

    for key in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
        value = adapted_observation.get(key)
        if value is not None:
            payload[key] = _batchify_optional_array(value)

    return payload


def resolve_openpi_policy_ref(policy_ref: str):
    spec = parse_openpi_policy_ref(policy_ref)
    policy_config_module, training_config_module = _import_openpi_modules(spec)
    train_config = training_config_module.get_config(spec.config)
    return policy_config_module.create_trained_policy(
        train_config,
        spec.checkpoint_dir,
        default_prompt=spec.default_prompt,
        pytorch_device=spec.pytorch_device,
    )


def resolve_openpi_hybrid_bridge_feed_builder(
    *,
    openpi_repo_root: str,
    config_name: str,
    checkpoint_dir: str,
    device: str,
    prefix_mode: str,
    export_precision: str,
    seed: int = 0,
) -> Callable[[dict[str, object], object], dict[str, np.ndarray]]:
    return resolve_openpi_hybrid_bridge_feed_builder_with_progress(
        openpi_repo_root=openpi_repo_root,
        config_name=config_name,
        checkpoint_dir=checkpoint_dir,
        device=device,
        prefix_mode=prefix_mode,
        export_precision=export_precision,
        seed=seed,
        on_progress=None,
    )


def resolve_openpi_mlx_hybrid_bridge_feed_builder(
    *,
    openpi_repo_root: str,
    config_name: str,
    checkpoint_dir: str,
    device: str,
    export_precision: str,
    seed: int = 0,
) -> Callable[[dict[str, object], object], dict[str, np.ndarray]]:
    cache_key = (
        str(Path(openpi_repo_root).resolve()),
        config_name,
        str(Path(checkpoint_dir).resolve()),
        device,
        export_precision,
    )
    if cache_key in _OPENPI_MLX_HYBRID_BRIDGE_FEED_BUILDER_CACHE:
        cached = _OPENPI_MLX_HYBRID_BRIDGE_FEED_BUILDER_CACHE[cache_key]
        if isinstance(cached, TimedFeedBuilder):
            cached.last_timings = {"feed_builder_cache_hit": 1.0}
        return cached

    timings: dict[str, float] = {}
    if cache_key in _OPENPI_MLX_HYBRID_BRIDGE_MODEL_CONTEXT_CACHE:
        model_module, observation_io_module, run_module, model, torch_device, args = _OPENPI_MLX_HYBRID_BRIDGE_MODEL_CONTEXT_CACHE[
            cache_key
        ]
        timings["model_context_cache_hit"] = 1.0
    else:
        model_module, observation_io_module, run_module = _import_openpi_mlx_hybrid_bridge_modules(openpi_repo_root)
        args = run_module.Args(
            config_name=config_name,
            checkpoint_dir=checkpoint_dir,
            device=device,
            seed=seed,
        )

        started = time.perf_counter()
        _, model = run_module._load_model(args)
        timings["load_model_s"] = round(time.perf_counter() - started, 6)
        torch_device = torch.device(device)

        started = time.perf_counter()
        model.paligemma_with_expert.to_bfloat16_for_selected_params("float32")
        model = model.float()
        timings["cast_model_s"] = round(time.perf_counter() - started, 6)

        started = time.perf_counter()
        model = model.to(torch_device)
        timings["move_model_s"] = round(time.perf_counter() - started, 6)

        started = time.perf_counter()
        model.eval()
        timings["eval_model_s"] = round(time.perf_counter() - started, 6)

        _OPENPI_MLX_HYBRID_BRIDGE_MODEL_CONTEXT_CACHE[cache_key] = (
            model_module,
            observation_io_module,
            run_module,
            model,
            torch_device,
            args,
        )

    def build_feed_impl(adapted_observation: dict[str, object], _runtime_config: object) -> dict[str, np.ndarray]:
        call_timings: dict[str, float] = {}
        payload = _adapted_observation_to_openpi_payload(adapted_observation)
        started = time.perf_counter()
        openpi_observation = model_module.Observation.from_dict(observation_io_module.to_torch(payload, torch_device))
        call_timings["payload_to_observation_s"] = round(time.perf_counter() - started, 6)

        started = time.perf_counter()
        feed, _meta = run_module._full_prefix_cache_mlx(model, openpi_observation, device=torch_device, seed=seed)
        call_timings["mlx_full_prefix_cache_s"] = round(time.perf_counter() - started, 6)

        started = time.perf_counter()
        if export_precision == "float16":
            result = {key: np.asarray(value, dtype=np.float16) for key, value in feed.items()}
        elif export_precision == "float32":
            result = {key: np.asarray(value, dtype=np.float32) for key, value in feed.items()}
        else:
            result = {key: np.asarray(value) for key, value in feed.items()}
        call_timings["numpy_export_s"] = round(time.perf_counter() - started, 6)
        builder.last_timings = call_timings
        return result

    builder = TimedFeedBuilder(build_fn=build_feed_impl, initialization_timings=timings)
    _OPENPI_MLX_HYBRID_BRIDGE_FEED_BUILDER_CACHE[cache_key] = builder
    return builder


def resolve_openpi_hybrid_bridge_feed_builder_with_progress(
    *,
    openpi_repo_root: str,
    config_name: str,
    checkpoint_dir: str,
    device: str,
    prefix_mode: str,
    export_precision: str,
    seed: int = 0,
    on_progress: Callable[[str, dict[str, float]], None] | None = None,
) -> Callable[[dict[str, object], object], dict[str, np.ndarray]]:
    cache_key = (
        str(Path(openpi_repo_root).resolve()),
        config_name,
        str(Path(checkpoint_dir).resolve()),
        device,
        prefix_mode or "observation",
        export_precision,
    )
    if cache_key in _OPENPI_HYBRID_BRIDGE_FEED_BUILDER_CACHE:
        if on_progress is not None:
            on_progress("feed_builder_cache_hit", {"cache_hit": 1.0})
        return _OPENPI_HYBRID_BRIDGE_FEED_BUILDER_CACHE[cache_key]
    timings: dict[str, float] = {}
    if cache_key in _OPENPI_HYBRID_BRIDGE_MODEL_CONTEXT_CACHE:
        model_module, observation_io_module, run_module, model, torch_device, args = _OPENPI_HYBRID_BRIDGE_MODEL_CONTEXT_CACHE[
            cache_key
        ]
        if on_progress is not None:
            on_progress("feed_builder_model_context_cache_hit", {"model_context_cache_hit": 1.0})
    else:
        model_module, observation_io_module, run_module = _import_openpi_hybrid_bridge_modules(openpi_repo_root)
        args = run_module.Args(
            config_name=config_name,
            checkpoint_dir=checkpoint_dir,
            device=device,
            prefix_mode=prefix_mode or "observation",
            export_precision=export_precision,
            seed=seed,
        )

        started = time.perf_counter()
        _, model = run_module._load_model(args)
        timings["load_model_s"] = round(time.perf_counter() - started, 6)
        if on_progress is not None:
            on_progress("feed_builder_model_loaded", timings)

        torch_device = torch.device(device)
        started = time.perf_counter()
        if export_precision == "float32":
            model.paligemma_with_expert.to_bfloat16_for_selected_params("float32")
            model = model.float()
        elif export_precision == "float16":
            model.paligemma_with_expert.to_bfloat16_for_selected_params("float16")
            model = model.half()
        timings["cast_model_s"] = round(time.perf_counter() - started, 6)
        if on_progress is not None:
            on_progress("feed_builder_model_cast", timings)

        started = time.perf_counter()
        model = model.to(torch_device)
        timings["move_model_s"] = round(time.perf_counter() - started, 6)
        if on_progress is not None:
            on_progress("feed_builder_model_moved", timings)

        started = time.perf_counter()
        model.eval()
        timings["eval_model_s"] = round(time.perf_counter() - started, 6)
        if on_progress is not None:
            on_progress("feed_builder_model_ready", timings)

        _OPENPI_HYBRID_BRIDGE_MODEL_CONTEXT_CACHE[cache_key] = (
            model_module,
            observation_io_module,
            run_module,
            model,
            torch_device,
            args,
        )

    def build_feed_impl(adapted_observation: dict[str, object], _runtime_config: object) -> dict[str, np.ndarray]:
        call_timings: dict[str, float] = {}
        payload = _adapted_observation_to_openpi_payload(adapted_observation)
        started = time.perf_counter()
        observation = model_module.Observation.from_dict(observation_io_module.to_torch(payload, torch_device))
        call_timings["payload_to_observation_s"] = round(time.perf_counter() - started, 6)

        started = time.perf_counter()
        images, img_masks, lang_tokens, lang_masks, state = run_module._prepare_prefix_inputs(
            model,
            observation,
            torch_device,
            args.prefix_mode,
        )
        call_timings["prepare_prefix_inputs_s"] = round(time.perf_counter() - started, 6)

        started = time.perf_counter()
        prefix_ctx = model.prepare_prefix(images, img_masks, lang_tokens, lang_masks, state)
        call_timings["prepare_prefix_s"] = round(time.perf_counter() - started, 6)

        started = time.perf_counter()
        flat_ctx = run_module._cast_flat_ctx(
            run_module.suffix_step_export.flatten_prefix_context(prefix_ctx),
            args.export_precision,
        )
        call_timings["flatten_prefix_context_s"] = round(time.perf_counter() - started, 6)

        started = time.perf_counter()
        torch.manual_seed(args.seed)
        x_init = model.sample_noise(
            (observation.state.shape[0], model.config.action_horizon, model.config.action_dim),
            torch_device,
        )
        if args.export_precision == "float32":
            x_init = x_init.to(dtype=torch.float32)
        elif args.export_precision == "float16":
            x_init = x_init.to(dtype=torch.float16)
        call_timings["sample_noise_s"] = round(time.perf_counter() - started, 6)

        started = time.perf_counter()
        packed_key_cache, packed_value_cache = flat_ctx.pack_kv_cache()
        call_timings["pack_kv_cache_s"] = round(time.perf_counter() - started, 6)

        started = time.perf_counter()
        feed = {
            "x_init": x_init.detach().cpu().numpy(),
            "state_workaround": flat_ctx.state.detach().cpu().numpy(),
            "key_cache_packed": packed_key_cache.detach().cpu().numpy(),
            "value_cache_packed": packed_value_cache.detach().cpu().numpy(),
        }
        call_timings["numpy_export_s"] = round(time.perf_counter() - started, 6)
        builder.last_timings = call_timings
        return feed

    builder = TimedFeedBuilder(build_fn=build_feed_impl, initialization_timings=timings)
    _OPENPI_HYBRID_BRIDGE_FEED_BUILDER_CACHE[cache_key] = builder
    return builder
