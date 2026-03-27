#!/usr/bin/env python

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import draccus

from lerobot.control_plane import IncidentAggregator, ReleaseRegistry, RolloutStatusBuilder, RolloutStatusStore
from lerobot.utils.utils import init_logging


@dataclass
class ControlPlaneRolloutReportConfig:
    registry_root: str = ".control_plane_registry"
    incident_root: str = ".edge_incidents"
    report_root: str = ".control_plane_reports"
    report_filename: str = "rollout_status.json"
    device_state_roots: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.registry_root:
            raise ValueError("registry_root cannot be empty.")
        if not self.incident_root:
            raise ValueError("incident_root cannot be empty.")
        if not self.report_root:
            raise ValueError("report_root cannot be empty.")
        if not self.report_filename:
            raise ValueError("report_filename cannot be empty.")


def generate_rollout_report(cfg: ControlPlaneRolloutReportConfig) -> Path:
    registry = ReleaseRegistry(Path(cfg.registry_root))
    aggregator = IncidentAggregator(Path(cfg.incident_root), registry)
    builder = RolloutStatusBuilder(
        registry=registry,
        incident_aggregator=aggregator,
        device_state_roots={device_id: Path(root) for device_id, root in cfg.device_state_roots.items()},
    )
    report = builder.build_report()
    store = RolloutStatusStore(Path(cfg.report_root))
    return store.write_report(report, filename=cfg.report_filename)


@draccus.wrap()
def main(cfg: ControlPlaneRolloutReportConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    output_path = generate_rollout_report(cfg)
    logging.info("Rollout report written to %s", output_path)


if __name__ == "__main__":
    main()
