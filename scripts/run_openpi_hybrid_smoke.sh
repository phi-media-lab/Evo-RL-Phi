#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OPENPI_REPO_ROOT="${OPENPI_REPO_ROOT:-$HOME/ane-openpi/openpi}"
OPENPI_SMOKE_VENV="${OPENPI_SMOKE_VENV:-$OPENPI_REPO_ROOT/.venv_evorl_smoke}"
OPENPI_CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:-$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch}"
OPENPI_MODEL_PATH="${OPENPI_MODEL_PATH:-$OPENPI_REPO_ROOT/artifacts/rollout8_pi0_base_bf16_aloha_obs_dynamic_packed_constaux_fp16.mlpackage}"
OPENPI_OBSERVATION_NPZ="${OPENPI_OBSERVATION_NPZ:-$OPENPI_REPO_ROOT/artifacts/observations/aloha_sim_row000000.npz}"
OPENPI_OBSERVATION_META_JSON="${OPENPI_OBSERVATION_META_JSON:-$OPENPI_REPO_ROOT/artifacts/observations/aloha_sim_row000000.json}"
OPENPI_ARTIFACT_ROOT="${OPENPI_ARTIFACT_ROOT:-/tmp/openpi_hybrid_edge_smoke_artifacts}"
OPENPI_SPOOL_ROOT="${OPENPI_SPOOL_ROOT:-/tmp/openpi_hybrid_edge_smoke_spool}"
OPENPI_POLICY_CONFIG="${OPENPI_POLICY_CONFIG:-pi0_aloha_sim}"
OPENPI_BRIDGE_MODE="${OPENPI_BRIDGE_MODE:-mlx_full_prefix_hostbridge}"
OPENPI_COMPUTE_UNIT="${OPENPI_COMPUTE_UNIT:-cpu_only}"
OPENPI_PREFIX_MODE="${OPENPI_PREFIX_MODE:-observation}"
OPENPI_EXPORT_PRECISION="${OPENPI_EXPORT_PRECISION:-float16}"
OPENPI_DEVICE="${OPENPI_DEVICE:-cpu}"
OPENPI_MAX_STEPS="${OPENPI_MAX_STEPS:-1}"
OPENPI_FPS="${OPENPI_FPS:-2}"

source "$OPENPI_SMOKE_VENV/bin/activate"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

python "$ROOT_DIR/src/lerobot/scripts/openpi_hybrid_edge_smoke.py" \
  --checkpoint-dir "$OPENPI_CHECKPOINT_DIR" \
  --model-path "$OPENPI_MODEL_PATH" \
  --observation-npz "$OPENPI_OBSERVATION_NPZ" \
  --observation-meta-json "$OPENPI_OBSERVATION_META_JSON" \
  --artifact-root "$OPENPI_ARTIFACT_ROOT" \
  --spool-root "$OPENPI_SPOOL_ROOT" \
  --policy-config "$OPENPI_POLICY_CONFIG" \
  --bridge-mode "$OPENPI_BRIDGE_MODE" \
  --compute-unit "$OPENPI_COMPUTE_UNIT" \
  --prefix-mode "$OPENPI_PREFIX_MODE" \
  --export-precision "$OPENPI_EXPORT_PRECISION" \
  --device "$OPENPI_DEVICE" \
  --openpi-repo-root "$OPENPI_REPO_ROOT" \
  --max-steps "$OPENPI_MAX_STEPS" \
  --fps "$OPENPI_FPS"
