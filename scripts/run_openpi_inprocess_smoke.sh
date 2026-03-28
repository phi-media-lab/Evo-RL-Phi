#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OPENPI_REPO_ROOT="${OPENPI_REPO_ROOT:-$HOME/ane-openpi/openpi}"
OPENPI_SMOKE_VENV="${OPENPI_SMOKE_VENV:-$OPENPI_REPO_ROOT/.venv_evorl_smoke}"
OPENPI_CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:-$HOME/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim.partial}"
OPENPI_OBSERVATION_NPZ="${OPENPI_OBSERVATION_NPZ:-$OPENPI_REPO_ROOT/artifacts/observations/aloha_sim_row000000.npz}"
OPENPI_OBSERVATION_META_JSON="${OPENPI_OBSERVATION_META_JSON:-$OPENPI_REPO_ROOT/artifacts/observations/aloha_sim_row000000.json}"
OPENPI_ARTIFACT_ROOT="${OPENPI_ARTIFACT_ROOT:-/tmp/openpi_edge_smoke_artifacts}"
OPENPI_SPOOL_ROOT="${OPENPI_SPOOL_ROOT:-/tmp/openpi_edge_smoke_spool}"
OPENPI_POLICY_CONFIG="${OPENPI_POLICY_CONFIG:-pi0_aloha_sim}"
OPENPI_MAX_STEPS="${OPENPI_MAX_STEPS:-2}"
OPENPI_FPS="${OPENPI_FPS:-2}"

source "$OPENPI_SMOKE_VENV/bin/activate"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

python "$ROOT_DIR/src/lerobot/scripts/openpi_edge_smoke.py" \
  --checkpoint-dir "$OPENPI_CHECKPOINT_DIR" \
  --observation-npz "$OPENPI_OBSERVATION_NPZ" \
  --observation-meta-json "$OPENPI_OBSERVATION_META_JSON" \
  --artifact-root "$OPENPI_ARTIFACT_ROOT" \
  --spool-root "$OPENPI_SPOOL_ROOT" \
  --policy-config "$OPENPI_POLICY_CONFIG" \
  --max-steps "$OPENPI_MAX_STEPS" \
  --fps "$OPENPI_FPS"
