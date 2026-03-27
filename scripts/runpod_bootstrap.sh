#!/usr/bin/env bash

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE_ROOT/Evo-RL}"
ENV_NAME="${ENV_NAME:-lerobot}"

echo "[runpod-bootstrap] workspace root: $WORKSPACE_ROOT"
echo "[runpod-bootstrap] repo dir: $REPO_DIR"
echo "[runpod-bootstrap] env name: $ENV_NAME"

mkdir -p "$WORKSPACE_ROOT/cache/huggingface"
mkdir -p "$WORKSPACE_ROOT/cache/pip"
mkdir -p "$WORKSPACE_ROOT/cache/uv"
mkdir -p "$WORKSPACE_ROOT/data/uploads"
mkdir -p "$WORKSPACE_ROOT/data/ingestion"
mkdir -p "$WORKSPACE_ROOT/data/materialized"
mkdir -p "$WORKSPACE_ROOT/outputs/train"
mkdir -p "$WORKSPACE_ROOT/outputs/eval"
mkdir -p "$WORKSPACE_ROOT/artifacts"
mkdir -p "$WORKSPACE_ROOT/logs"
mkdir -p "$WORKSPACE_ROOT/logs/wandb"

if ! command -v conda >/dev/null 2>&1; then
  echo "[runpod-bootstrap] conda not found"
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi

conda activate "$ENV_NAME"

export HF_HOME="$WORKSPACE_ROOT/cache/huggingface"
export PIP_CACHE_DIR="$WORKSPACE_ROOT/cache/pip"
export WANDB_DIR="$WORKSPACE_ROOT/logs/wandb"
export TOKENIZERS_PARALLELISM=false

cat > "$WORKSPACE_ROOT/runpod_env.sh" <<EOF
export HF_HOME=$WORKSPACE_ROOT/cache/huggingface
export PIP_CACHE_DIR=$WORKSPACE_ROOT/cache/pip
export WANDB_DIR=$WORKSPACE_ROOT/logs/wandb
export TOKENIZERS_PARALLELISM=false
EOF

cd "$REPO_DIR"
pip install --upgrade pip
pip install -e .

python -c "import torch, lerobot; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
python -m compileall src/lerobot/edge src/lerobot/control_plane src/lerobot/cloud

echo "[runpod-bootstrap] done"
echo "[runpod-bootstrap] source $WORKSPACE_ROOT/runpod_env.sh"
