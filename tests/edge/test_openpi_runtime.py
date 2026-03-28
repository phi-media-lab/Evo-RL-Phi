#!/usr/bin/env python

from __future__ import annotations

import json

import numpy as np
import torch

import lerobot.edge.openpi_runtime as openpi_runtime_module
import lerobot.edge.openpi_loader as openpi_loader_module
from lerobot.edge.openpi_runtime import (
    OpenPICoreMLRuntime,
    OpenPICoreMLRuntimeConfig,
    OpenPICoreMLRuntimeFactory,
    OpenPIHybridBridgeRuntime,
    OpenPIHybridBridgeRuntimeConfig,
    OpenPIHybridBridgeRuntimeFactory,
    OpenPIInProcessRuntime,
    OpenPIInProcessRuntimeConfig,
    OpenPIInProcessRuntimeFactory,
    OpenPIObservationAdapter,
    OpenPIObservationAdapterConfig,
    OpenPIWebsocketRuntime,
    OpenPIWebsocketRuntimeConfig,
)
from lerobot.edge.runner import EdgeRobotRunner
from lerobot.edge.spool import EdgeEpisodeSpool
from lerobot.robots.utils import make_robot_from_config
from tests.mocks.mock_robot import MockRobotConfig


class FakeOpenPIClient:
    def __init__(self):
        self.calls: list[dict] = []
        self.reset_calls = 0

    def infer(self, obs: dict) -> dict:
        self.calls.append(obs)
        return {
            "actions": np.asarray(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
                dtype=np.float32,
            )
        }

    def reset(self) -> None:
        self.reset_calls += 1


class FakeOpenPIPolicy:
    def __init__(self):
        self.calls: list[dict] = []
        self.reset_calls = 0

    def infer(self, obs: dict) -> dict:
        self.calls.append(obs)
        return {
            "actions": np.asarray(
                [
                    [0.7, 0.8, 0.9],
                    [1.0, 1.1, 1.2],
                ],
                dtype=np.float32,
            )
        }

    def reset(self) -> None:
        self.reset_calls += 1


class FakeOpenPIHybridPredictor:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, feed: dict) -> dict:
        self.calls.append(feed)
        return {
            "actions": np.asarray(
                [
                    [0.25, 0.5, 0.75],
                    [1.0, 1.25, 1.5],
                ],
                dtype=np.float32,
            )
        }


class FakeOpenPICoreMLPredictor:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, feed: dict) -> dict:
        self.calls.append(feed)
        return {
            "actions": np.asarray(
                [
                    [0.15, 0.3, 0.45],
                    [0.6, 0.75, 0.9],
                ],
                dtype=np.float32,
            )
        }


class FakeOpenPIRolloutPredictor:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, feed: dict) -> dict:
        self.calls.append(feed)
        rollout = np.zeros((1, 2, 32), dtype=np.float32)
        rollout[0, 0, :14] = np.arange(14, dtype=np.float32)
        rollout[0, 1, :14] = np.arange(14, 28, dtype=np.float32)
        return {"rollout": rollout}


def test_openpi_observation_adapter_maps_robot_state():
    adapter = OpenPIObservationAdapter(
        OpenPIObservationAdapterConfig(
            state_keys=["motor_1.pos", "motor_2.pos", "motor_3.pos"],
            prompt="pick up cube",
        )
    )

    adapted = adapter.adapt(
        {
            "motor_1.pos": 1.0,
            "motor_2.pos": 2.0,
            "motor_3.pos": 3.0,
        }
    )

    assert adapted["prompt"] == "pick up cube"
    np.testing.assert_allclose(adapted["state"], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))


def test_openpi_observation_adapter_maps_aloha_raw_inputs():
    adapter = OpenPIObservationAdapter(
        OpenPIObservationAdapterConfig(
            mode="aloha_raw",
            state_key="state",
            state_dim=14,
            image_keys=[
                "image.base_0_rgb",
                "image.left_wrist_0_rgb",
                "image.right_wrist_0_rgb",
            ],
            image_aliases={
                "image.base_0_rgb": "cam_high",
                "image.left_wrist_0_rgb": "cam_left_wrist",
                "image.right_wrist_0_rgb": "cam_right_wrist",
            },
            prompt_key="prompt",
            transpose_images_to_chw=True,
        )
    )

    adapted = adapter.adapt(
        {
            "state": np.arange(32, dtype=np.float32).reshape(1, 32),
            "image.base_0_rgb": np.arange(1 * 4 * 5 * 3, dtype=np.uint8).reshape(1, 4, 5, 3),
            "image.left_wrist_0_rgb": np.ones((1, 4, 5, 3), dtype=np.uint8),
            "image.right_wrist_0_rgb": np.full((1, 4, 5, 3), 2, dtype=np.uint8),
            "prompt": "Transfer cube",
        }
    )

    assert adapted["prompt"] == "Transfer cube"
    np.testing.assert_allclose(adapted["state"], np.arange(14, dtype=np.float32))
    assert sorted(adapted["images"]) == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    assert adapted["images"]["cam_high"].shape == (3, 4, 5)
    np.testing.assert_array_equal(
        adapted["images"]["cam_high"],
        np.transpose(np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3), (2, 0, 1)),
    )


def test_openpi_observation_adapter_maps_openpi_raw_inputs():
    adapter = OpenPIObservationAdapter(
        OpenPIObservationAdapterConfig(
            mode="openpi_raw",
            state_key="state",
            image_keys=[
                "image.base_0_rgb",
                "image.left_wrist_0_rgb",
            ],
            prompt_key="prompt",
        )
    )

    adapted = adapter.adapt(
        {
            "state": np.arange(32, dtype=np.float32).reshape(1, 32),
            "image.base_0_rgb": np.arange(1 * 4 * 5 * 3, dtype=np.uint8).reshape(1, 4, 5, 3),
            "image.left_wrist_0_rgb": np.ones((1, 4, 5, 3), dtype=np.uint8),
            "prompt": "Transfer cube",
        }
    )

    assert adapted["prompt"] == "Transfer cube"
    assert adapted["state"].shape == (1, 32)
    assert sorted(adapted["image"]) == ["base_0_rgb", "left_wrist_0_rgb"]
    assert adapted["image"]["base_0_rgb"].shape == (1, 4, 5, 3)
    np.testing.assert_array_equal(adapted["image_mask"]["base_0_rgb"], np.asarray([True]))


def test_openpi_observation_adapter_preserves_tokenized_prompt_fields():
    adapter = OpenPIObservationAdapter(
        OpenPIObservationAdapterConfig(
            mode="aloha_raw",
            state_key="state",
            state_dim=14,
            image_keys=["image.base_0_rgb"],
            image_aliases={"image.base_0_rgb": "cam_high"},
            transpose_images_to_chw=True,
        )
    )

    adapted = adapter.adapt(
        {
            "state": np.arange(14, dtype=np.float32),
            "image.base_0_rgb": np.zeros((4, 5, 3), dtype=np.uint8),
            "tokenized_prompt": np.asarray([[1, 2, 3]], dtype=np.int32),
            "tokenized_prompt_mask": np.asarray([[True, True, False]]),
        }
    )

    np.testing.assert_array_equal(adapted["tokenized_prompt"], np.asarray([1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(adapted["tokenized_prompt_mask"], np.asarray([True, True, False]))


def test_openpi_runtime_feeds_edge_runner_and_spool(tmp_path):
    fake_client = FakeOpenPIClient()
    runtime = OpenPIWebsocketRuntime(
        OpenPIWebsocketRuntimeConfig(
            server_uri="ws://127.0.0.1:8000",
            action_horizon=2,
            action_dim=3,
            observation=OpenPIObservationAdapterConfig(
                state_keys=["motor_1.pos", "motor_2.pos", "motor_3.pos"],
                prompt="stack blocks",
            ),
        ),
        client_factory=lambda cfg: fake_client,
    )
    robot = make_robot_from_config(
        MockRobotConfig(
            n_motors=3,
            random_values=False,
            static_values=[1.0, 2.0, 3.0],
        )
    )
    spool = EdgeEpisodeSpool(tmp_path / "spool")

    robot.connect()
    try:
        runner = EdgeRobotRunner(
            robot=robot,
            runtime=runtime,
            spool=spool,
            policy_artifact_id="openpi-ws-artifact",
            processor_bundle_id="openpi-adapter-v1",
            task_id="stack",
            channel="dev",
            action_processor=None,
        )
        result = runner.run_episode(max_steps=2, fps=20)
    finally:
        robot.disconnect()

    assert result.step_count == 2
    assert fake_client.reset_calls == 1
    assert len(fake_client.calls) == 1
    np.testing.assert_allclose(fake_client.calls[0]["state"], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert fake_client.calls[0]["prompt"] == "stack blocks"

    sealed_episode_dir = tmp_path / "spool" / "sealed" / result.episode_id
    with (sealed_episode_dir / "steps.jsonl").open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    np.testing.assert_allclose(
        [lines[0]["action"]["motor_1.pos"], lines[0]["action"]["motor_2.pos"], lines[0]["action"]["motor_3.pos"]],
        [0.1, 0.2, 0.3],
    )
    np.testing.assert_allclose(
        [lines[1]["action"]["motor_1.pos"], lines[1]["action"]["motor_2.pos"], lines[1]["action"]["motor_3.pos"]],
        [0.4, 0.5, 0.6],
    )


def test_openpi_in_process_runtime_feeds_edge_runner_and_spool(tmp_path):
    fake_policy = FakeOpenPIPolicy()
    runtime = OpenPIInProcessRuntime(
        OpenPIInProcessRuntimeConfig(
            action_horizon=2,
            action_dim=3,
            policy_ref="fake/pi0",
            observation=OpenPIObservationAdapterConfig(
                state_keys=["motor_1.pos", "motor_2.pos", "motor_3.pos"],
                prompt="push button",
            ),
        ),
        policy=fake_policy,
    )
    robot = make_robot_from_config(
        MockRobotConfig(
            n_motors=3,
            random_values=False,
            static_values=[1.0, 2.0, 3.0],
        )
    )
    spool = EdgeEpisodeSpool(tmp_path / "spool")

    robot.connect()
    try:
        runner = EdgeRobotRunner(
            robot=robot,
            runtime=runtime,
            spool=spool,
            policy_artifact_id="openpi-inproc-artifact",
            processor_bundle_id="openpi-adapter-v1",
            task_id="push",
            channel="dev",
            action_processor=None,
        )
        result = runner.run_episode(max_steps=2, fps=20)
    finally:
        robot.disconnect()

    assert result.step_count == 2
    assert fake_policy.reset_calls == 1
    assert len(fake_policy.calls) == 1
    np.testing.assert_allclose(fake_policy.calls[0]["state"], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert fake_policy.calls[0]["prompt"] == "push button"

    sealed_episode_dir = tmp_path / "spool" / "sealed" / result.episode_id
    with (sealed_episode_dir / "steps.jsonl").open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    np.testing.assert_allclose(
        [lines[0]["action"]["motor_1.pos"], lines[0]["action"]["motor_2.pos"], lines[0]["action"]["motor_3.pos"]],
        [0.7, 0.8, 0.9],
    )
    np.testing.assert_allclose(
        [lines[1]["action"]["motor_1.pos"], lines[1]["action"]["motor_2.pos"], lines[1]["action"]["motor_3.pos"]],
        [1.0, 1.1, 1.2],
    )


def test_openpi_in_process_runtime_factory_resolves_policy():
    fake_policy = FakeOpenPIPolicy()
    factory = OpenPIInProcessRuntimeFactory(policy_resolver=lambda policy_ref: fake_policy)

    runtime = factory.build_runtime(
        OpenPIInProcessRuntimeConfig(
            action_horizon=2,
            action_dim=3,
            policy_ref="fake/pi0",
            observation=OpenPIObservationAdapterConfig(
                state_keys=["motor_1.pos", "motor_2.pos", "motor_3.pos"],
            ),
        )
    )

    assert isinstance(runtime, OpenPIInProcessRuntime)
    assert factory.supports_runtime(runtime)


def test_openpi_hybrid_runtime_feeds_edge_runner_and_spool(tmp_path):
    fake_predictor = FakeOpenPIHybridPredictor()
    runtime = OpenPIHybridBridgeRuntime(
        OpenPIHybridBridgeRuntimeConfig(
            model_path="/tmp/model.mlpackage",
            bridge_mode="mlx_full_prefix_hostbridge",
            action_horizon=2,
            action_dim=3,
            observation=OpenPIObservationAdapterConfig(
                state_keys=["motor_1.pos", "motor_2.pos", "motor_3.pos"],
                prompt="hybrid pick",
            ),
            feed_contract_path="/tmp/feed_contract.json",
            prefix_mode="mlx_prefix_v1",
        ),
        predictor=fake_predictor,
    )
    robot = make_robot_from_config(
        MockRobotConfig(
            n_motors=3,
            random_values=False,
            static_values=[1.0, 2.0, 3.0],
        )
    )
    spool = EdgeEpisodeSpool(tmp_path / "spool")

    robot.connect()
    try:
        runner = EdgeRobotRunner(
            robot=robot,
            runtime=runtime,
            spool=spool,
            policy_artifact_id="openpi-hybrid-artifact",
            processor_bundle_id="openpi-hybrid-adapter-v1",
            task_id="pick",
            channel="dev",
            action_processor=None,
        )
        result = runner.run_episode(max_steps=2, fps=20)
    finally:
        robot.disconnect()

    assert result.step_count == 2
    assert len(fake_predictor.calls) == 1
    assert fake_predictor.calls[0]["bridge_mode"] == "mlx_full_prefix_hostbridge"
    np.testing.assert_allclose(fake_predictor.calls[0]["observation"]["state"], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))

    sealed_episode_dir = tmp_path / "spool" / "sealed" / result.episode_id
    with (sealed_episode_dir / "steps.jsonl").open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    np.testing.assert_allclose(
        [lines[0]["action"]["motor_1.pos"], lines[0]["action"]["motor_2.pos"], lines[0]["action"]["motor_3.pos"]],
        [0.25, 0.5, 0.75],
    )
    assert "feed_builder_total_s" in lines[0]["info"]
    assert "predictor_total_s" in lines[0]["info"]


def test_openpi_hybrid_runtime_factory_resolves_predictor():
    fake_predictor = FakeOpenPIHybridPredictor()
    factory = OpenPIHybridBridgeRuntimeFactory(predictor_factory=lambda runtime_config: fake_predictor)

    runtime = factory.build_runtime(
        OpenPIHybridBridgeRuntimeConfig(
            model_path="/tmp/model.mlpackage",
            bridge_mode="hostbridge_v1",
            action_horizon=2,
            action_dim=3,
            observation=OpenPIObservationAdapterConfig(
                state_keys=["motor_1.pos", "motor_2.pos", "motor_3.pos"],
            ),
        )
    )

    assert isinstance(runtime, OpenPIHybridBridgeRuntime)
    assert factory.supports_runtime(runtime)


def test_openpi_hybrid_runtime_decodes_single_rollout_output():
    fake_predictor = FakeOpenPIRolloutPredictor()
    runtime = OpenPIHybridBridgeRuntime(
        OpenPIHybridBridgeRuntimeConfig(
            model_path="/tmp/model.mlpackage",
            bridge_mode="mlx_full_prefix_hostbridge",
            action_horizon=2,
            action_dim=14,
            observation=OpenPIObservationAdapterConfig(
                state_keys=["motor_1.pos", "motor_2.pos", "motor_3.pos"],
            ),
            model_output_key="rollout",
            output_transform="aloha_actions",
        ),
        predictor=fake_predictor,
    )

    chunk = runtime.refill_action_queue(
        {
            "motor_1.pos": 1.0,
            "motor_2.pos": 2.0,
            "motor_3.pos": 3.0,
        }
    )

    expected = openpi_runtime_module._encode_aloha_actions_from_openpi_internal(  # noqa: SLF001
        torch.as_tensor(
            np.asarray(
                [
                    np.arange(14, dtype=np.float32),
                    np.arange(14, 28, dtype=np.float32),
                ]
            )
        )
    )
    np.testing.assert_allclose(chunk.numpy(), expected.numpy())


def test_openpi_coreml_runtime_decodes_single_rollout_output():
    fake_predictor = FakeOpenPIRolloutPredictor()
    runtime = OpenPICoreMLRuntime(
        OpenPICoreMLRuntimeConfig(
            model_path="/tmp/model.mlpackage",
            action_horizon=2,
            action_dim=14,
            observation=OpenPIObservationAdapterConfig(
                state_keys=["motor_1.pos", "motor_2.pos", "motor_3.pos"],
            ),
            model_output_key="rollout",
            output_transform="aloha_actions",
        ),
        predictor=fake_predictor,
    )

    chunk = runtime.refill_action_queue(
        {
            "motor_1.pos": 1.0,
            "motor_2.pos": 2.0,
            "motor_3.pos": 3.0,
        }
    )

    expected = openpi_runtime_module._encode_aloha_actions_from_openpi_internal(  # noqa: SLF001
        torch.as_tensor(
            np.asarray(
                [
                    np.arange(14, dtype=np.float32),
                    np.arange(14, 28, dtype=np.float32),
                ]
            )
        )
    )
    np.testing.assert_allclose(chunk.numpy(), expected.numpy())


def test_openpi_hybrid_runtime_reports_coreml_input_shape_mismatch():
    runtime = OpenPIHybridBridgeRuntime(
        OpenPIHybridBridgeRuntimeConfig(
            model_path="/tmp/model.mlpackage",
            bridge_mode="mlx_full_prefix_hostbridge",
            action_horizon=2,
            action_dim=14,
            observation=OpenPIObservationAdapterConfig(
                state_keys=["motor_1.pos", "motor_2.pos", "motor_3.pos"],
            ),
        ),
        predictor=lambda feed: (_ for _ in ()).throw(
            ValueError("CoreML input shape mismatch for 'state_workaround': expected (1, 32), got (1, 14)")
        ),
        feed_builder=lambda adapted_observation, runtime_config: {
            "x_init": np.zeros((1, 50, 32), dtype=np.float16),
            "state_workaround": np.zeros((1, 14), dtype=np.float16),
            "key_cache_packed": np.zeros((18, 1, 1, 816, 256), dtype=np.float16),
            "value_cache_packed": np.zeros((18, 1, 1, 816, 256), dtype=np.float16),
        },
    )

    try:
        runtime.refill_action_queue(
            {
                "motor_1.pos": 1.0,
                "motor_2.pos": 2.0,
                "motor_3.pos": 3.0,
            }
        )
    except ValueError as exc:
        assert "state_workaround" in str(exc)
        assert "(1, 32)" in str(exc)
    else:
        raise AssertionError("Expected input shape mismatch to raise ValueError.")


def test_openpi_hybrid_runtime_factory_uses_real_feed_builder_when_checkpoint_config_present(monkeypatch):
    feed_builder_calls: list[dict] = []

    def fake_real_feed_builder(**kwargs):
        def build_feed(adapted_observation, runtime_config):
            feed_builder_calls.append(
                {
                    "kwargs": kwargs,
                    "state": adapted_observation["state"],
                    "config_name": runtime_config.config_name,
                }
            )
            return {"actions": np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)}

        return build_feed

    monkeypatch.setattr(openpi_runtime_module, "resolve_openpi_hybrid_bridge_feed_builder", fake_real_feed_builder)
    factory = OpenPIHybridBridgeRuntimeFactory(predictor_factory=lambda runtime_config: (lambda feed: feed))
    runtime = factory.build_runtime(
        OpenPIHybridBridgeRuntimeConfig(
            model_path="/tmp/model.mlpackage",
            bridge_mode="hostbridge_v1",
            action_horizon=2,
            action_dim=3,
            observation=OpenPIObservationAdapterConfig(
                state_keys=["motor_1.pos", "motor_2.pos", "motor_3.pos"],
            ),
            config_name="pi0_aloha_sim",
            checkpoint_dir="/tmp/pi0_aloha_sim",
            prefix_mode="observation",
            export_precision="float16",
            device="cpu",
        )
    )

    action = runtime.pop_next_action(
        {
            "motor_1.pos": 1.0,
            "motor_2.pos": 2.0,
            "motor_3.pos": 3.0,
        }
    )

    np.testing.assert_allclose(action.numpy(), np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    assert len(feed_builder_calls) == 1
    assert feed_builder_calls[0]["kwargs"]["config_name"] == "pi0_aloha_sim"
    assert feed_builder_calls[0]["kwargs"]["checkpoint_dir"] == "/tmp/pi0_aloha_sim"
    np.testing.assert_allclose(feed_builder_calls[0]["state"], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))


def test_openpi_hybrid_runtime_factory_uses_mlx_feed_builder_for_mlx_bridge_mode(monkeypatch):
    feed_builder_calls: list[dict[str, object]] = []

    def fake_mlx_feed_builder(**kwargs):
        def build_feed(adapted_observation, runtime_config):
            feed_builder_calls.append(
                {
                    "kwargs": kwargs,
                    "state_shape": np.asarray(adapted_observation["state"]).shape,
                    "bridge_mode": runtime_config.bridge_mode,
                }
            )
            return {"actions": np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)}

        return build_feed

    monkeypatch.setattr(openpi_runtime_module, "resolve_openpi_mlx_hybrid_bridge_feed_builder", fake_mlx_feed_builder)
    factory = OpenPIHybridBridgeRuntimeFactory(predictor_factory=lambda runtime_config: (lambda feed: feed))
    runtime = factory.build_runtime(
        OpenPIHybridBridgeRuntimeConfig(
            model_path="/tmp/model.mlpackage",
            bridge_mode="mlx_full_prefix_hostbridge",
            action_horizon=2,
            action_dim=3,
            observation=OpenPIObservationAdapterConfig(
                mode="openpi_raw",
                state_key="state",
                image_keys=["image.base_0_rgb"],
            ),
            config_name="pi0_aloha_sim",
            checkpoint_dir="/tmp/pi0_aloha_sim",
            prefix_mode="observation",
            export_precision="float16",
            device="cpu",
        )
    )

    action = runtime.pop_next_action(
        {
            "state": np.arange(32, dtype=np.float32).reshape(1, 32),
            "image.base_0_rgb": np.zeros((1, 4, 5, 3), dtype=np.uint8),
        }
    )

    np.testing.assert_allclose(action.numpy(), np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    assert len(feed_builder_calls) == 1
    assert feed_builder_calls[0]["kwargs"]["config_name"] == "pi0_aloha_sim"
    assert feed_builder_calls[0]["kwargs"]["checkpoint_dir"] == "/tmp/pi0_aloha_sim"
    assert feed_builder_calls[0]["bridge_mode"] == "mlx_full_prefix_hostbridge"
    assert feed_builder_calls[0]["state_shape"] == (1, 32)


def test_openpi_hybrid_bridge_feed_builder_caches_loaded_builder(monkeypatch):
    openpi_loader_module._OPENPI_HYBRID_BRIDGE_FEED_BUILDER_CACHE.clear()
    openpi_loader_module._OPENPI_HYBRID_BRIDGE_MODEL_CONTEXT_CACHE.clear()
    fake_model_module = object()
    fake_observation_io_module = object()

    class FakePaliGemmaWithExpert:
        def to_bfloat16_for_selected_params(self, precision):
            return None

    class FakeModel:
        def __init__(self):
            self.paligemma_with_expert = FakePaliGemmaWithExpert()

        def float(self):
            return self

        def half(self):
            return self

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeRunModule:
        class Args:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class suffix_step_export:
            @staticmethod
            def flatten_prefix_context(prefix_ctx):
                return prefix_ctx

        @staticmethod
        def _load_model(args):
            return object(), FakeModel()

        @staticmethod
        def _prepare_prefix_inputs(model, observation, device, prefix_mode):
            return [], [], [], [], np.zeros((1, 14), dtype=np.float32)

        @staticmethod
        def _cast_flat_ctx(flat_ctx, export_precision):
            return flat_ctx

    monkeypatch.setattr(
        openpi_loader_module,
        "_import_openpi_hybrid_bridge_modules",
        lambda root: (fake_model_module, fake_observation_io_module, FakeRunModule),
    )

    progress_events: list[str] = []
    first_builder = openpi_loader_module.resolve_openpi_hybrid_bridge_feed_builder_with_progress(
        openpi_repo_root="/tmp/openpi",
        config_name="pi0_aloha_sim",
        checkpoint_dir="/tmp/pi0_aloha_sim_pytorch",
        device="cpu",
        prefix_mode="observation",
        export_precision="float16",
        on_progress=lambda stage, timings: progress_events.append(stage),
    )
    second_builder = openpi_loader_module.resolve_openpi_hybrid_bridge_feed_builder_with_progress(
        openpi_repo_root="/tmp/openpi",
        config_name="pi0_aloha_sim",
        checkpoint_dir="/tmp/pi0_aloha_sim_pytorch",
        device="cpu",
        prefix_mode="observation",
        export_precision="float16",
        on_progress=lambda stage, timings: progress_events.append(stage),
    )

    assert first_builder is second_builder
    assert "feed_builder_model_loaded" in progress_events
    assert "feed_builder_model_ready" in progress_events
    assert progress_events[-1] == "feed_builder_cache_hit"

    openpi_loader_module._OPENPI_HYBRID_BRIDGE_FEED_BUILDER_CACHE.clear()
    third_builder = openpi_loader_module.resolve_openpi_hybrid_bridge_feed_builder_with_progress(
        openpi_repo_root="/tmp/openpi",
        config_name="pi0_aloha_sim",
        checkpoint_dir="/tmp/pi0_aloha_sim_pytorch",
        device="cpu",
        prefix_mode="observation",
        export_precision="float16",
        on_progress=lambda stage, timings: progress_events.append(stage),
    )
    assert third_builder is not None
    assert "feed_builder_model_context_cache_hit" in progress_events


def test_openpi_coreml_runtime_collects_predictor_timings():
    class FakeTimedPredictor:
        def __init__(self):
            self.initialization_timings = {"load_compiled_model_s": 0.5}
            self.last_timings = None

        def __call__(self, feed):
            self.last_timings = {"validate_feed_s": 0.1, "predict_s": 0.2}
            return {"actions": np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)}

    runtime = OpenPICoreMLRuntime(
        OpenPICoreMLRuntimeConfig(
            model_path="/tmp/model.mlpackage",
            action_horizon=2,
            action_dim=3,
            observation=OpenPIObservationAdapterConfig(
                state_keys=["motor_1.pos", "motor_2.pos", "motor_3.pos"],
            ),
        ),
        predictor=FakeTimedPredictor(),
        feed_builder=lambda adapted_observation, runtime_config: {"actions": np.zeros((2, 3), dtype=np.float32)},
    )

    runtime.refill_action_queue(
        {
            "motor_1.pos": 1.0,
            "motor_2.pos": 2.0,
            "motor_3.pos": 3.0,
        }
    )

    assert runtime.last_runtime_info["predictor_init_load_compiled_model_s"] == 0.5
    assert runtime.last_runtime_info["predictor_validate_feed_s"] == 0.1
    assert runtime.last_runtime_info["predictor_predict_s"] == 0.2


def test_openpi_coreml_runtime_feeds_edge_runner_and_spool(tmp_path):
    fake_predictor = FakeOpenPICoreMLPredictor()
    runtime = OpenPICoreMLRuntime(
        OpenPICoreMLRuntimeConfig(
            model_path="/tmp/model.mlpackage",
            action_horizon=2,
            action_dim=3,
            observation=OpenPIObservationAdapterConfig(
                state_keys=["motor_1.pos", "motor_2.pos", "motor_3.pos"],
                prompt="coreml pick",
            ),
        ),
        predictor=fake_predictor,
    )
    robot = make_robot_from_config(
        MockRobotConfig(
            n_motors=3,
            random_values=False,
            static_values=[1.0, 2.0, 3.0],
        )
    )
    spool = EdgeEpisodeSpool(tmp_path / "spool")

    robot.connect()
    try:
        runner = EdgeRobotRunner(
            robot=robot,
            runtime=runtime,
            spool=spool,
            policy_artifact_id="openpi-coreml-artifact",
            processor_bundle_id="openpi-coreml-adapter-v1",
            task_id="pick",
            channel="dev",
            action_processor=None,
        )
        result = runner.run_episode(max_steps=2, fps=20)
    finally:
        robot.disconnect()

    assert result.step_count == 2
    assert len(fake_predictor.calls) == 1
    np.testing.assert_allclose(fake_predictor.calls[0]["observation"]["state"], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))

    sealed_episode_dir = tmp_path / "spool" / "sealed" / result.episode_id
    with (sealed_episode_dir / "steps.jsonl").open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    np.testing.assert_allclose(
        [lines[0]["action"]["motor_1.pos"], lines[0]["action"]["motor_2.pos"], lines[0]["action"]["motor_3.pos"]],
        [0.15, 0.3, 0.45],
    )


def test_openpi_coreml_runtime_factory_resolves_predictor():
    fake_predictor = FakeOpenPICoreMLPredictor()
    factory = OpenPICoreMLRuntimeFactory(predictor_factory=lambda runtime_config: fake_predictor)

    runtime = factory.build_runtime(
        OpenPICoreMLRuntimeConfig(
            model_path="/tmp/model.mlpackage",
            action_horizon=2,
            action_dim=3,
            observation=OpenPIObservationAdapterConfig(
                state_keys=["motor_1.pos", "motor_2.pos", "motor_3.pos"],
            ),
        )
    )

    assert isinstance(runtime, OpenPICoreMLRuntime)
    assert factory.supports_runtime(runtime)
