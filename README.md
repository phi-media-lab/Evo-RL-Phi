# Evo-RL

[中文说明](./README.zh-CN.md)

Evo-RL is a LeRobot-based real-world RL codebase focused on a practical cloud-edge workflow:

- train and evaluate policies in the cloud
- package deployable artifacts
- execute policies locally on the edge robot
- record episodes durably
- upload structured data back to the control plane
- manage rollout, rollback, and incident reporting by channel

This repository is currently beyond the “architecture-only” stage. It already contains a working minimum skeleton for:

- local edge execution
- episode spool and upload
- HTTP ingestion and materialization
- training smoke and artifact build
- artifact and channel-based rollout
- boundary-based model switching
- model-crash rollback
- incident aggregation
- control-plane rollout reporting
- persistent cloud stack orchestration

## Current Status

Implemented today:

- edge runtime contracts, episode schema, artifact manifest, release schema
- local runtime, watchdog, runner, recorder, spool, uploader
- filesystem-backed and HTTP-backed ingestion
- filesystem-backed and HTTP-backed materializer
- materialization to local `LeRobotDataset`
- one-step training smoke and artifact builder
- release registry and device/channel mapping
- model manager with active, pending, and previous-active artifact state
- episode-boundary deployment loop
- rollback on `model_crash`
- incident sink on edge
- control-plane incident aggregation
- rollout status snapshot builder and report script
- auto-release daemon and unified `cloud_stack`

Not implemented yet:

- production-grade auth and storage backends
- long-running training jobs with real eval gates
- production-grade approval and promotion workflow
- remote health reporting and production audit pipeline

Detailed project plan and phase tracking live in [DRAFT.md](./DRAFT.md).
Current edge runtime completion status lives in [EDGE_STACK_STATUS.md](./EDGE_STACK_STATUS.md).
OpenPI local runtime packaging and migration notes live in [OPENPI_LOCAL_RUNTIME.md](./OPENPI_LOCAL_RUNTIME.md).

## Repository Shape

Key modules:

- [src/lerobot/edge](./src/lerobot/edge)
  Local runtime, deployment loop, recorder, spool, uploader, watchdog, incidents.
- [src/lerobot/control_plane](./src/lerobot/control_plane)
  Artifact schema, registry, incident aggregation, rollout status reporting.
- [src/lerobot/cloud](./src/lerobot/cloud)
  Ingestion, materializer, artifact build, and cloud-side helpers.
- [src/lerobot/scripts/edge_run_local.py](./src/lerobot/scripts/edge_run_local.py)
  Edge execution entrypoint.
- [src/lerobot/scripts/cloud_stack.py](./src/lerobot/scripts/cloud_stack.py)
  Unified cloud stack entrypoint.
- [src/lerobot/scripts/control_plane_rollout_report.py](./src/lerobot/scripts/control_plane_rollout_report.py)
  Control-plane rollout report entrypoint.

## Environment Setup

Recommended local environment:

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
pip install -e .
```

Then verify the package imports:

```bash
python -c "import lerobot; print(lerobot.__version__)"
```

Project-specific agent instructions for local work are in [AGENTS.md](./AGENTS.md).

## Main Entry Points

### Edge Execution

Run the local edge loop with:

```bash
python -m lerobot.scripts.edge_run_local --help
```

This path supports:

- dry-run execution
- local spool and optional upload
- device/channel-based artifact resolution
- episode-boundary target sync
- incident recording

Core implementation:

- [src/lerobot/edge/runtime.py](./src/lerobot/edge/runtime.py)
- [src/lerobot/edge/runner.py](./src/lerobot/edge/runner.py)
- [src/lerobot/edge/deployment_loop.py](./src/lerobot/edge/deployment_loop.py)
- [src/lerobot/edge/model_manager.py](./src/lerobot/edge/model_manager.py)

### Control Plane Rollout Report

Generate a rollout snapshot from registry, incidents, and device state:

```bash
python -m lerobot.scripts.control_plane_rollout_report --help
```

This path reads:

- registry state
- edge incident files
- edge model `state.json` files

And writes a report that summarizes:

- channel target artifact
- assigned devices
- active/pending/previous-active artifact state
- incident and rollback counts

Core implementation:

- [src/lerobot/control_plane/registry.py](./src/lerobot/control_plane/registry.py)
- [src/lerobot/control_plane/incidents.py](./src/lerobot/control_plane/incidents.py)
- [src/lerobot/control_plane/rollout.py](./src/lerobot/control_plane/rollout.py)

### Unified Cloud Stack

Run the combined ingestion, materializer, and auto-release stack with:

```bash
python -m lerobot.scripts.cloud_stack --help
```

This path supports:

- HTTP ingestion
- HTTP materialization
- auto release on new materialized data
- persistent `tmux`/`systemd` deployment assets
- `/healthz` and `/status` endpoints

Core implementation:

- [src/lerobot/scripts/cloud_stack.py](./src/lerobot/scripts/cloud_stack.py)
- [src/lerobot/scripts/control_plane_auto_release_daemon.py](./src/lerobot/scripts/control_plane_auto_release_daemon.py)
- [src/lerobot/cloud/ingestion.py](./src/lerobot/cloud/ingestion.py)
- [src/lerobot/cloud/materializer.py](./src/lerobot/cloud/materializer.py)

## Tests

The current edge/control-plane skeleton is covered by focused tests under [tests/edge](./tests/edge):

- `test_edge_runner.py`
- `test_edge_uploader.py`
- `test_edge_run_local.py`
- `test_release_registry.py`
- `test_model_manager_sync.py`
- `test_deployment_loop.py`
- `test_incident_aggregator.py`
- `test_rollout_status.py`
- `test_control_plane_rollout_report.py`
- `test_http_ingestion.py`
- `test_materializer.py`
- `test_http_materializer.py`
- `test_artifact_builder.py`
- `test_train_and_build.py`
- `test_release_controller.py`
- `test_release_cycle.py`
- `test_auto_release_controller.py`
- `test_cloud_stack.py`

Run the current regression suite with:

```bash
pytest -q tests/edge/test_control_plane_rollout_report.py \
  tests/edge/test_rollout_status.py \
  tests/edge/test_incident_aggregator.py \
  tests/edge/test_deployment_loop.py \
  tests/edge/test_model_manager_sync.py \
  tests/edge/test_edge_run_local.py \
  tests/edge/test_release_registry.py \
  tests/edge/test_edge_runner.py \
  tests/edge/test_edge_uploader.py
```

## Development Workflow

Recommended git workflow:

1. Keep the original repository as upstream.
2. Push active development to your fork.
3. Use topic branches for coherent slices.
4. Merge back through PRs by subsystem, not as one giant diff.

Suggested PR split:

- contracts + schemas
- edge runtime + spool + uploader
- registry + deployment loop + rollback
- incidents + rollout reporting
- docs and runbooks

## Practical Roadmap

The next high-value work is not more edge abstraction. It is:

1. implement a real ingestion service
2. materialize uploaded episodes into training-ready datasets
3. connect edge-returned data into the existing training pipeline
4. build real deployment artifacts from checkpoints
5. move the release loop from local files to a service-backed controller

## Project Plan

For the implementation plan, current phase status, and remaining gaps, see:

- [DRAFT.md](./DRAFT.md)
