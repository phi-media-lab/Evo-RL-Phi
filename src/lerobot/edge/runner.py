#!/usr/bin/env python

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.processor import EnvTransition, RobotAction, RobotObservation, TransitionKey, create_transition
from lerobot.robots.robot import Robot

from .contracts import EdgeRuntimeContract, default_edge_runtime_contract
from .episode import EdgeEpisodeRecord
from .recorder import EdgeEpisodeRecorder
from .runtime_protocol import EdgePolicyRuntime
from .spool import EdgeEpisodeSpool
from .watchdog import EdgeWatchdog, WatchdogIncident


@dataclass
class EdgeRunnerResult:
    episode_id: str
    step_count: int
    duration_s: float
    stop_reason: str = "completed"
    watchdog_incident: WatchdogIncident | None = None


class EdgeRobotRunner:
    """Minimal single-robot edge loop with local policy inference and durable recording."""

    def __init__(
        self,
        *,
        robot: Robot,
        runtime: EdgePolicyRuntime,
        spool: EdgeEpisodeSpool,
        policy_artifact_id: str,
        processor_bundle_id: str,
        task_id: str,
        channel: str | None = None,
        action_processor: Any | None = None,
        runtime_contract: EdgeRuntimeContract | None = None,
    ):
        self.robot = robot
        self.runtime = runtime
        self.spool = spool
        self.policy_artifact_id = policy_artifact_id
        self.processor_bundle_id = processor_bundle_id
        self.task_id = task_id
        self.channel = channel
        self.action_processor = action_processor
        self.runtime_contract = runtime_contract or default_edge_runtime_contract()
        self.watchdog = EdgeWatchdog(self.runtime_contract)

    def run_episode(self, max_steps: int, fps: int) -> EdgeRunnerResult:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if fps <= 0:
            raise ValueError("fps must be positive.")

        episode = EdgeEpisodeRecord(
            robot_id=str(self.robot.id),
            robot_type=self.robot.name,
            policy_artifact_id=self.policy_artifact_id,
            processor_bundle_id=self.processor_bundle_id,
            task_id=self.task_id,
            channel=self.channel,
        )
        recorder = EdgeEpisodeRecorder(spool=self.spool, episode=episode)

        self.runtime.reset()
        self.watchdog.reset()
        start_t = time.perf_counter()
        stop_reason = "completed"
        watchdog_incident: WatchdogIncident | None = None

        for step_idx in range(max_steps):
            loop_start_t = time.perf_counter()
            raw_observation = self.robot.get_observation()

            inference_start_t = time.perf_counter()
            try:
                action_tensor = self.runtime.pop_next_action(raw_observation)
            except Exception as error:
                watchdog_incident = self.watchdog.model_crash(error)
                stop_reason = watchdog_incident.code
                break
            inference_duration_ms = (time.perf_counter() - inference_start_t) * 1000.0
            watchdog_incident = self.watchdog.observe_inference_duration_ms(inference_duration_ms)
            if watchdog_incident is not None and watchdog_incident.action == "safe_stop":
                stop_reason = watchdog_incident.code
                break

            robot_action, processed_transition = self._process_action(action_tensor, raw_observation)
            sent_action = self.robot.send_action(robot_action)

            info = dict(processed_transition.get(TransitionKey.INFO, {}) or {})
            complementary_data = processed_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
            if complementary_data:
                info["complementary_data"] = complementary_data
            info["runtime_inference_ms"] = inference_duration_ms
            info["runtime_queue_size"] = self.runtime.queue_size
            runtime_stage_info = getattr(self.runtime, "last_runtime_info", None)
            if isinstance(runtime_stage_info, dict):
                info.update(runtime_stage_info)

            done_local = bool(processed_transition.get(TransitionKey.DONE, False))
            recorder.append_step(
                observation=raw_observation,
                action=sent_action,
                teleop_override=bool(info.get("intervened", False) or info.get("is_intervention", False)),
                reward_raw=float(processed_transition.get(TransitionKey.REWARD, 0.0) or 0.0),
                done_local=done_local,
                done_train=done_local,
                info=info,
            )

            if done_local:
                stop_reason = "done_local"
                break

            elapsed_s = time.perf_counter() - loop_start_t
            stale_incident = self.watchdog.observe_stale_action_latency_ms(elapsed_s * 1000.0)
            if stale_incident is not None and stale_incident.action == "safe_stop":
                watchdog_incident = stale_incident
                stop_reason = stale_incident.code
                break
            period_s = 1.0 / fps
            if elapsed_s < period_s:
                time.sleep(period_s - elapsed_s)

        duration_s = time.perf_counter() - start_t
        recorder.seal(
            summary={
                "step_count": recorder.step_idx,
                "duration_s": duration_s,
                "policy_artifact_id": self.policy_artifact_id,
                "processor_bundle_id": self.processor_bundle_id,
                "stop_reason": stop_reason,
                "watchdog_incident": None if watchdog_incident is None else watchdog_incident.__dict__,
            }
        )
        return EdgeRunnerResult(
            episode_id=episode.episode_id,
            step_count=recorder.step_idx,
            duration_s=duration_s,
            stop_reason=stop_reason,
            watchdog_incident=watchdog_incident,
        )

    def _process_action(
        self,
        action_tensor: torch.Tensor,
        observation: RobotObservation,
    ) -> tuple[RobotAction, EnvTransition]:
        transition = create_transition(observation=observation, action=action_tensor)
        if self.action_processor is None:
            robot_action = self._tensor_to_robot_action(action_tensor)
            transition[TransitionKey.ACTION] = robot_action
            return robot_action, transition

        processed_transition = self.action_processor(transition)
        robot_action = processed_transition.get(TransitionKey.ACTION)
        if not isinstance(robot_action, dict):
            raise TypeError("Processed action must be a RobotAction dictionary.")
        return robot_action, processed_transition

    def _tensor_to_robot_action(self, action_tensor: torch.Tensor) -> RobotAction:
        action_tensor = action_tensor.detach().cpu().reshape(-1)
        action_names = list(self.robot.action_features.keys())
        if len(action_names) != action_tensor.numel():
            raise ValueError(
                f"Action dimension mismatch: robot expects {len(action_names)} values, "
                f"but runtime produced {action_tensor.numel()}."
            )
        return {name: float(action_tensor[idx]) for idx, name in enumerate(action_names)}
