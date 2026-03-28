# OpenPI Local Runtime

This document describes the minimal local packaging needed to run the current `Evo-RL + ane-openpi` inference stack on another Mac, such as a higher-spec MacBook Air.

## Scope

The current integration supports these OpenPI runtime families inside Evo-RL:

- `in_process`
- `websocket`
- `coreml`
- `hybrid_bridge_v1`

The most advanced validated path today is:

`OpenPI artifact -> EdgeModelManager.activate_artifact() -> OpenPIHybridBridgeRuntime -> EdgeRobotRunner -> spool`

with `bridge_mode=mlx_full_prefix_hostbridge`.

## Recommended Workspace Layout

Use a single local workspace with stable paths:

```text
<workspace>/
  Evo-RL/
  ane-openpi/openpi/
  checkpoints/
  artifacts/
  samples/
```

Recommended environment variables:

```bash
export OPENPI_REPO_ROOT="$HOME/ane-openpi/openpi"
export OPENPI_SMOKE_VENV="$OPENPI_REPO_ROOT/.venv_evorl_smoke"
export OPENPI_CHECKPOINT_DIR="$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch"
export OPENPI_MODEL_PATH="$OPENPI_REPO_ROOT/artifacts/rollout8_pi0_base_bf16_aloha_obs_dynamic_packed_constaux_fp16.mlpackage"
export OPENPI_OBSERVATION_NPZ="$OPENPI_REPO_ROOT/artifacts/observations/aloha_sim_row000000.npz"
export OPENPI_OBSERVATION_META_JSON="$OPENPI_REPO_ROOT/artifacts/observations/aloha_sim_row000000.json"
```

## Setup

Bootstrap the smoke environment with:

```bash
bash scripts/setup_openpi_edge_env.sh
```

This script:

- creates `OPENPI_SMOKE_VENV` if needed
- installs the OpenPI smoke dependencies
- installs `openpi-client`
- installs `ane-openpi`
- installs `mlx` / `mlx-metal`
- copies `transformers_replace` into the venv `transformers` package

## In-Process Smoke

Run:

```bash
bash scripts/run_openpi_inprocess_smoke.sh
```

This verifies:

- `policy_ref -> resolve_openpi_policy_ref(...)`
- real OpenPI policy load
- `OpenPIInProcessRuntime`
- `EdgeRobotRunner`
- spool write

Default outputs:

- artifact root: `/tmp/openpi_edge_smoke_artifacts`
- spool root: `/tmp/openpi_edge_smoke_spool`

## Hybrid Smoke

Run:

```bash
bash scripts/run_openpi_hybrid_smoke.sh
```

This verifies:

- OpenPI artifact build
- `EdgeModelManager.activate_artifact()`
- `OpenPIHybridBridgeRuntime`
- MLX prefix feed construction
- CoreML rollout predictor
- `EdgeRobotRunner`
- spool write

Default outputs:

- artifact root: `/tmp/openpi_hybrid_edge_smoke_artifacts`
- spool root: `/tmp/openpi_hybrid_edge_smoke_spool`

## Current Status

What is already proven:

- real `pi0_aloha_sim` in-process load and infer
- real artifact activation path for OpenPI
- real `mlx_full_prefix_hostbridge` hybrid edge smoke reaches spool
- current integration works inside the existing Evo-RL edge stack

What is still weak:

- first-step latency for hybrid is still very high
- the dominant cost is in first-time runtime preparation, not the edge shell itself
- runtime stage timing has been added to `steps.jsonl`, but a full timed hybrid run may still take minutes

## Performance Notes

For hybrid runs, inspect the first step in `steps.jsonl` and look for:

- `runtime_inference_ms`
- `feed_builder_total_s`
- `feed_builder_init_*`
- `feed_builder_*`
- `predictor_total_s`
- `predictor_init_*`
- `predictor_*`

These fields are the current source of truth for where the first-step latency goes.

## Migration Guidance

Before moving to a stronger Mac:

1. Copy both codebases without changing relative structure.
2. Copy the converted PyTorch checkpoint.
3. Copy the rollout `mlpackage`.
4. Copy the sample observation files.
5. Re-run `bash scripts/setup_openpi_edge_env.sh`.
6. Re-run both smoke scripts unchanged.

If the stronger machine still shows poor first-step latency, the next optimization target is:

- persistent runtime reuse
- model/predictor preloading
- cross-episode reuse of model context and compiled predictor state
