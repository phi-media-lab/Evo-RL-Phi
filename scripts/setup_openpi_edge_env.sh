#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OPENPI_REPO_ROOT="${OPENPI_REPO_ROOT:-$HOME/ane-openpi/openpi}"
OPENPI_SMOKE_VENV="${OPENPI_SMOKE_VENV:-$OPENPI_REPO_ROOT/.venv_evorl_smoke}"
OPENPI_VENV_PYTHON="${OPENPI_VENV_PYTHON:-}"

if [ -z "$OPENPI_VENV_PYTHON" ]; then
  if command -v python3.11 >/dev/null 2>&1; then
    OPENPI_VENV_PYTHON="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    OPENPI_VENV_PYTHON="$(command -v python3)"
  else
    echo "Missing python3.11 and python3." >&2
    exit 1
  fi
fi

echo "ROOT_DIR=$ROOT_DIR"
echo "OPENPI_REPO_ROOT=$OPENPI_REPO_ROOT"
echo "OPENPI_SMOKE_VENV=$OPENPI_SMOKE_VENV"
echo "OPENPI_VENV_PYTHON=$OPENPI_VENV_PYTHON"

if [ ! -d "$OPENPI_REPO_ROOT" ]; then
  echo "Missing OPENPI_REPO_ROOT: $OPENPI_REPO_ROOT" >&2
  exit 1
fi

if [ ! -d "$OPENPI_SMOKE_VENV" ]; then
  "$OPENPI_VENV_PYTHON" -m venv "$OPENPI_SMOKE_VENV"
fi

source "$OPENPI_SMOKE_VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

python -m pip install \
  "torch==2.7.1" \
  "jax==0.5.3" \
  "jaxlib==0.5.3" \
  "flax==0.10.2" \
  "transformers==4.53.2" \
  "orbax-checkpoint==0.11.13" \
  "numpy<2" \
  websockets \
  pytest \
  chex==0.1.89 \
  tqdm-loggable \
  numpydantic \
  flatbuffers \
  opencv-python \
  polars \
  coremltools \
  mlx \
  mlx-metal

python -m pip install -e "$OPENPI_REPO_ROOT/packages/openpi-client"
python -m pip install -e "$OPENPI_REPO_ROOT" --no-deps

PATCH_SRC="$OPENPI_REPO_ROOT/src/openpi/models_pytorch/transformers_replace"
OPENPI_TRANSFORMERS_SITE_PACKAGES="$(python - <<'PY'
from pathlib import Path
import transformers
print(Path(transformers.__file__).resolve().parent)
PY
)"
if [ -d "$PATCH_SRC" ] && [ -d "$OPENPI_TRANSFORMERS_SITE_PACKAGES" ]; then
  cp -R "$PATCH_SRC"/. "$OPENPI_TRANSFORMERS_SITE_PACKAGES"/
fi

echo
echo "Environment ready."
echo "Activate with:"
echo "  source \"$OPENPI_SMOKE_VENV/bin/activate\""
