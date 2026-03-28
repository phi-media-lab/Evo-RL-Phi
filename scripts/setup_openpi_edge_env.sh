#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OPENPI_REPO_ROOT="${OPENPI_REPO_ROOT:-$HOME/ane-openpi/openpi}"
OPENPI_SMOKE_VENV="${OPENPI_SMOKE_VENV:-$OPENPI_REPO_ROOT/.venv_evorl_smoke}"
OPENPI_TRANSFORMERS_SITE_PACKAGES="${OPENPI_TRANSFORMERS_SITE_PACKAGES:-$OPENPI_SMOKE_VENV/lib/python3.11/site-packages/transformers}"

echo "ROOT_DIR=$ROOT_DIR"
echo "OPENPI_REPO_ROOT=$OPENPI_REPO_ROOT"
echo "OPENPI_SMOKE_VENV=$OPENPI_SMOKE_VENV"

if [ ! -d "$OPENPI_REPO_ROOT" ]; then
  echo "Missing OPENPI_REPO_ROOT: $OPENPI_REPO_ROOT" >&2
  exit 1
fi

if [ ! -d "$OPENPI_SMOKE_VENV" ]; then
  python3.11 -m venv "$OPENPI_SMOKE_VENV"
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
python -m pip install -e "$OPENPI_REPO_ROOT"

PATCH_SRC="$OPENPI_REPO_ROOT/src/openpi/models_pytorch/transformers_replace"
if [ -d "$PATCH_SRC" ] && [ -d "$OPENPI_TRANSFORMERS_SITE_PACKAGES" ]; then
  cp -R "$PATCH_SRC"/. "$OPENPI_TRANSFORMERS_SITE_PACKAGES"/
fi

echo
echo "Environment ready."
echo "Activate with:"
echo "  source \"$OPENPI_SMOKE_VENV/bin/activate\""
