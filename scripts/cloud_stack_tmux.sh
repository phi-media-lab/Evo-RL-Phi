#!/usr/bin/env bash

set -euo pipefail

ACTION="${1:-status}"
SESSION_NAME="${SESSION_NAME:-evorl-cloud-stack}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE_ROOT/Evo-RL}"
ENV_NAME="${ENV_NAME:-lerobot}"

HOST="${HOST:-127.0.0.1}"
INGESTION_PORT="${INGESTION_PORT:-8000}"
MATERIALIZER_PORT="${MATERIALIZER_PORT:-8001}"
STATUS_PORT="${STATUS_PORT:-8002}"

INGESTION_ROOT="${INGESTION_ROOT:-$WORKSPACE_ROOT/data/ingestion}"
MATERIALIZED_ROOT="${MATERIALIZED_ROOT:-$WORKSPACE_ROOT/data/materialized}"
DATASET_ROOT="${DATASET_ROOT:-$WORKSPACE_ROOT/data/dataset}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-$WORKSPACE_ROOT/outputs/train}"
ARTIFACT_OUTPUT_ROOT="${ARTIFACT_OUTPUT_ROOT:-$WORKSPACE_ROOT/artifacts}"
REGISTRY_ROOT="${REGISTRY_ROOT:-$WORKSPACE_ROOT/data/registry}"
STATE_ROOT="${STATE_ROOT:-$WORKSPACE_ROOT/data/controller_state}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$WORKSPACE_ROOT/logs/cloud_stack}"
INCIDENT_ROOT="${INCIDENT_ROOT:-$WORKSPACE_ROOT/data/incidents}"
REPORT_ROOT="${REPORT_ROOT:-$WORKSPACE_ROOT/data/reports}"
CHANNEL="${CHANNEL:-staging}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-artifact-cloud-stack}"
ROBOT_TYPE="${ROBOT_TYPE:-mock_robot}"
CAMERA_LAYOUT="${CAMERA_LAYOUT:-single_arm_mock}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-5}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
ROLLOUT_REASON="${ROLLOUT_REASON:-cloud stack release}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[cloud-stack-tmux] tmux not found"
  exit 1
fi

STACK_CMD=$(
  cat <<EOF
cd "$REPO_DIR"
if [ -f "$WORKSPACE_ROOT/runpod_env.sh" ]; then source "$WORKSPACE_ROOT/runpod_env.sh"; fi
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
python -m lerobot.scripts.cloud_stack \
  --host "$HOST" \
  --ingestion-port "$INGESTION_PORT" \
  --materializer-port "$MATERIALIZER_PORT" \
  --status-port "$STATUS_PORT" \
  --ingestion-root "$INGESTION_ROOT" \
  --materialized-root "$MATERIALIZED_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --train-output-root "$TRAIN_OUTPUT_ROOT" \
  --artifact-output-root "$ARTIFACT_OUTPUT_ROOT" \
  --registry-root "$REGISTRY_ROOT" \
  --state-root "$STATE_ROOT" \
  --runtime-root "$RUNTIME_ROOT" \
  --incident-root "$INCIDENT_ROOT" \
  --report-root "$REPORT_ROOT" \
  --channel "$CHANNEL" \
  --artifact-prefix "$ARTIFACT_PREFIX" \
  --robot-type "$ROBOT_TYPE" \
  --camera-layout "$CAMERA_LAYOUT" \
  --rollout-reason "$ROLLOUT_REASON" \
  --poll-interval-s "$POLL_INTERVAL_S" \
  ${MAX_ITERATIONS:+--max-iterations "$MAX_ITERATIONS"}
EOF
)

case "$ACTION" in
  start)
    mkdir -p "$RUNTIME_ROOT"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "[cloud-stack-tmux] session already exists: $SESSION_NAME"
      exit 0
    fi
    tmux new-session -d -s "$SESSION_NAME" "$STACK_CMD"
    echo "[cloud-stack-tmux] started session: $SESSION_NAME"
    echo "[cloud-stack-tmux] status: http://$HOST:$STATUS_PORT/status"
    echo "[cloud-stack-tmux] logs: $RUNTIME_ROOT/daemon.log"
    ;;
  stop)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux kill-session -t "$SESSION_NAME"
      echo "[cloud-stack-tmux] stopped session: $SESSION_NAME"
    else
      echo "[cloud-stack-tmux] session not running: $SESSION_NAME"
    fi
    ;;
  status)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "[cloud-stack-tmux] session running: $SESSION_NAME"
      echo "[cloud-stack-tmux] status: http://$HOST:$STATUS_PORT/status"
      tmux list-panes -t "$SESSION_NAME" -F "pane_pid=#{pane_pid} pane_current_command=#{pane_current_command}"
    else
      echo "[cloud-stack-tmux] session not running: $SESSION_NAME"
      exit 1
    fi
    ;;
  logs)
    if [ -f "$RUNTIME_ROOT/daemon.log" ]; then
      tail -n "${TAIL_LINES:-40}" "$RUNTIME_ROOT/daemon.log"
    else
      echo "[cloud-stack-tmux] log file not found: $RUNTIME_ROOT/daemon.log"
      exit 1
    fi
    ;;
  attach)
    tmux attach -t "$SESSION_NAME"
    ;;
  *)
    echo "usage: $0 {start|stop|status|logs|attach}"
    exit 1
    ;;
esac
