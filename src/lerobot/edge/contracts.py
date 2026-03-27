#!/usr/bin/env python

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EdgeWatchdogPolicy:
    """Static watchdog policy for the first edge runtime iteration."""

    consecutive_inference_timeouts_before_safe_stop: int = 3
    stale_action_threshold_ms: int = 150
    action_queue_low_watermark: int = 2
    single_forward_timeout_action: str = "hold_position"
    repeated_forward_timeout_action: str = "safe_stop"
    camera_timeout_action: str = "safe_stop"
    empty_action_queue_action: str = "hold_position"
    model_crash_action: str = "rollback_or_safe_stop"

    def __post_init__(self) -> None:
        if self.consecutive_inference_timeouts_before_safe_stop <= 0:
            raise ValueError("consecutive_inference_timeouts_before_safe_stop must be positive.")
        if self.stale_action_threshold_ms <= 0:
            raise ValueError("stale_action_threshold_ms must be positive.")
        if self.action_queue_low_watermark < 0:
            raise ValueError("action_queue_low_watermark must be non-negative.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EdgeRuntimeContract:
    """Execution-time contract shared by the edge runtime, watchdog, and model manager."""

    runtime_abi_version: str = "v1"
    observation_schema_version: str = "v1"
    action_schema_version: str = "v1"
    control_frequency_hz: int = 20
    target_inference_budget_ms: int = 25
    hard_inference_timeout_ms: int = 40
    switch_only_at_episode_boundary: bool = True
    allow_switch_in_idle_state: bool = True
    allow_switch_in_maintenance_mode: bool = True
    watchdog: EdgeWatchdogPolicy = EdgeWatchdogPolicy()

    def __post_init__(self) -> None:
        if self.control_frequency_hz <= 0:
            raise ValueError("control_frequency_hz must be positive.")
        if not self.runtime_abi_version:
            raise ValueError("runtime_abi_version cannot be empty.")
        if not self.observation_schema_version:
            raise ValueError("observation_schema_version cannot be empty.")
        if not self.action_schema_version:
            raise ValueError("action_schema_version cannot be empty.")
        if self.target_inference_budget_ms <= 0:
            raise ValueError("target_inference_budget_ms must be positive.")
        if self.hard_inference_timeout_ms < self.target_inference_budget_ms:
            raise ValueError("hard_inference_timeout_ms must be >= target_inference_budget_ms.")

    @property
    def control_period_ms(self) -> float:
        return 1000.0 / self.control_frequency_hz

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["control_period_ms"] = self.control_period_ms
        return payload


def default_edge_runtime_contract() -> EdgeRuntimeContract:
    return EdgeRuntimeContract()
