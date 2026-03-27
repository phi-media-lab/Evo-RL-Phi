#!/usr/bin/env python

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .release import DeviceChannelAssignment, ReleaseChannelState


@dataclass(frozen=True)
class ReleaseRegistryLayout:
    root: Path
    channels_dirname: str = "channels"
    devices_dirname: str = "devices"

    @property
    def channels(self) -> Path:
        return self.root / self.channels_dirname

    @property
    def devices(self) -> Path:
        return self.root / self.devices_dirname


class ReleaseRegistry:
    """Filesystem-backed channel target and device assignment registry."""

    def __init__(self, root: str | Path):
        self.layout = ReleaseRegistryLayout(root=Path(root))
        self.layout.channels.mkdir(parents=True, exist_ok=True)
        self.layout.devices.mkdir(parents=True, exist_ok=True)

    def set_channel_target(
        self,
        *,
        channel: str,
        target_artifact_id: str,
        rollout_reason: str | None = None,
    ) -> ReleaseChannelState:
        state = ReleaseChannelState(
            channel=channel,
            target_artifact_id=target_artifact_id,
            rollout_reason=rollout_reason,
        )
        self._write_json(self.layout.channels / f"{channel}.json", state.to_dict())
        return state

    def get_channel_target(self, channel: str) -> ReleaseChannelState | None:
        path = self.layout.channels / f"{channel}.json"
        if not path.exists():
            return None
        return ReleaseChannelState(**self._read_json(path))

    def list_channel_targets(self) -> list[ReleaseChannelState]:
        return [
            ReleaseChannelState(**self._read_json(path))
            for path in sorted(self.layout.channels.glob("*.json"))
        ]

    def assign_device(
        self,
        *,
        device_id: str,
        channel: str,
        robot_id: str | None = None,
        task_id: str | None = None,
    ) -> DeviceChannelAssignment:
        assignment = DeviceChannelAssignment(
            device_id=device_id,
            channel=channel,
            robot_id=robot_id,
            task_id=task_id,
        )
        self._write_json(self.layout.devices / f"{device_id}.json", assignment.to_dict())
        return assignment

    def get_device_assignment(self, device_id: str) -> DeviceChannelAssignment | None:
        path = self.layout.devices / f"{device_id}.json"
        if not path.exists():
            return None
        return DeviceChannelAssignment(**self._read_json(path))

    def list_device_assignments(self) -> list[DeviceChannelAssignment]:
        return [
            DeviceChannelAssignment(**self._read_json(path))
            for path in sorted(self.layout.devices.glob("*.json"))
        ]

    def resolve_target_for_device(
        self,
        device_id: str,
        *,
        fallback_channel: str | None = None,
    ) -> ReleaseChannelState:
        assignment = self.get_device_assignment(device_id)
        channel = fallback_channel
        if assignment is not None:
            channel = assignment.channel
        if channel is None:
            raise ValueError(f"No channel assignment found for device '{device_id}'.")

        channel_state = self.get_channel_target(channel)
        if channel_state is None:
            raise ValueError(f"No target artifact configured for channel '{channel}'.")
        return channel_state

    def _write_json(self, path: Path, payload: dict) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _read_json(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
