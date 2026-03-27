#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass

from .contracts import EdgeRuntimeContract


@dataclass(frozen=True)
class WatchdogIncident:
    code: str
    action: str
    message: str


class EdgeWatchdog:
    """Evaluates loop health against the frozen edge runtime contract."""

    def __init__(self, contract: EdgeRuntimeContract):
        self.contract = contract
        self._consecutive_inference_timeouts = 0

    def reset(self) -> None:
        self._consecutive_inference_timeouts = 0

    def observe_inference_duration_ms(self, duration_ms: float) -> WatchdogIncident | None:
        if duration_ms <= self.contract.hard_inference_timeout_ms:
            self._consecutive_inference_timeouts = 0
            return None

        self._consecutive_inference_timeouts += 1
        if (
            self._consecutive_inference_timeouts
            >= self.contract.watchdog.consecutive_inference_timeouts_before_safe_stop
        ):
            return WatchdogIncident(
                code="inference_timeout_repeated",
                action=self.contract.watchdog.repeated_forward_timeout_action,
                message=(
                    f"Inference exceeded hard timeout {self.contract.hard_inference_timeout_ms}ms "
                    f"for {self._consecutive_inference_timeouts} consecutive iterations."
                ),
            )

        return WatchdogIncident(
            code="inference_timeout_once",
            action=self.contract.watchdog.single_forward_timeout_action,
            message=f"Inference exceeded hard timeout {self.contract.hard_inference_timeout_ms}ms.",
        )

    def observe_action_queue_size(self, queue_size: int) -> WatchdogIncident | None:
        if queue_size > 0:
            return None
        return WatchdogIncident(
            code="action_queue_empty",
            action=self.contract.watchdog.empty_action_queue_action,
            message="Action queue is empty after attempting refill.",
        )

    def observe_stale_action_latency_ms(self, latency_ms: float) -> WatchdogIncident | None:
        if latency_ms <= self.contract.watchdog.stale_action_threshold_ms:
            return None
        return WatchdogIncident(
            code="stale_action",
            action=self.contract.watchdog.single_forward_timeout_action,
            message=(
                f"Action latency {latency_ms:.2f}ms exceeded stale-action threshold "
                f"{self.contract.watchdog.stale_action_threshold_ms}ms."
            ),
        )

    def camera_timeout(self) -> WatchdogIncident:
        return WatchdogIncident(
            code="camera_timeout",
            action=self.contract.watchdog.camera_timeout_action,
            message="Robot observation acquisition timed out.",
        )

    def model_crash(self, error: Exception) -> WatchdogIncident:
        return WatchdogIncident(
            code="model_crash",
            action=self.contract.watchdog.model_crash_action,
            message=f"Model runtime crashed: {error}",
        )
