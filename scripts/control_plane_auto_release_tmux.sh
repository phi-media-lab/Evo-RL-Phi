#!/usr/bin/env bash

set -euo pipefail

ACTION="${1:-status}"
SESSION_NAME="${SESSION_NAME:-evorl-control-plane}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE_ROOT/Evo-RL}"
ENV_NAME="${ENV_NAME:-lerobot}"

MATERIALIZED_ROOT="${MATERIALIZED_ROOT:-$WORKSPACE_ROOT/data/materialized}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-$WORKSPACE_ROOT/outputs/train}"
ARTIFACT_OUTPUT_ROOT="${ARTIFACT_OUTPUT_ROOT:-$WORKSPACE_ROOT/artifacts}"
REGISTRY_ROOT="${REGISTRY_ROOT:-$WORKSPACE_ROOT/data/registry}"
STATE_ROOT="${STATE_ROOT:-$WORKSPACE_ROOT/data/controller_state}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$WORKSPACE_ROOT/logs/control_plane_auto_release}"
INCIDENT_ROOT="${INCIDENT_ROOT:-$WORKSPACE_ROOT/data/incidents}"
REPORT_ROOT="${REPORT_ROOT:-$WORKSPACE_ROOT/data/reports}"
CHANNEL="${CHANNEL:-staging}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-artifact-auto-release}"
ROBOT_TYPE="${ROBOT_TYPE:-mock_robot}"
CAMERA_LAYOUT="${CAMERA_LAYOUT:-single_arm_mock}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-5}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
ROLLOUT_REASON="${ROLLOUT_REASON:-auto release daemon}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[auto-release-tmux] tmux not found"
  exit 1
fi

DAEMON_CMD=$(
  cat <<EOF
cd "$REPO_DIR"
if [ -f "$WORKSPACE_ROOT/runpod_env.sh" ]; then source "$WORKSPACE_ROOT/runpod_env.sh"; fi
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
python -m lerobot.scripts.control_plane_auto_release_daemon \
  --materialized-root "$MATERIALIZED_ROOT" \
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
      echo "[auto-release-tmux] session already exists: $SESSION_NAME"
      exit 0
    fi
    tmux new-session -d -s "$SESSION_NAME" "$DAEMON_CMD"
    echo "[auto-release-tmux] started session: $SESSION_NAME"
    echo "[auto-release-tmux] logs: $RUNTIME_ROOT/daemon.log"
    ;;
  stop)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux kill-session -t "$SESSION_NAME"
      echo "[auto-release-tmux] stopped session: $SESSION_NAME"
    else
      echo "[auto-release-tmux] session not running: $SESSION_NAME"
    fi
    ;;
  status)
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "[auto-release-tmux] session running: $SESSION_NAME"
      tmux list-panes -t "$SESSION_NAME" -F "pane_pid=#{pane_pid} pane_current_command=#{pane_current_command}"
    else
      echo "[auto-release-tmux] session not running: $SESSION_NAME"
      exit 1
    fi
    ;;
  logs)
    if [ -f "$RUNTIME_ROOT/daemon.log" ]; then
      tail -n "${TAIL_LINES:-40}" "$RUNTIME_ROOT/daemon.log"
    else
      echo "[auto-release-tmux] log file not found: $RUNTIME_ROOT/daemon.log"
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
