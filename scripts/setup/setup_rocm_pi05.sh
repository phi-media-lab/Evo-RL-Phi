#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${EVORL_VENV_DIR:-$REPO_ROOT/.venvs/pi05-openpi-ssp}"
OPENPI_DIR="${OPENPI_DIR:-$REPO_ROOT/../openpi}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SITE_PACKAGES="$VENV_DIR/lib/python3.12/site-packages"

echo "[setup] repo root: $REPO_ROOT"
echo "[setup] venv dir:  $VENV_DIR"
echo "[setup] openpi dir: $OPENPI_DIR"

if [ ! -d "$OPENPI_DIR" ]; then
  echo "[setup] missing OpenPI source at: $OPENPI_DIR" >&2
  echo "[setup] clone it first, for example:" >&2
  echo "git clone --depth=1 --recurse-submodules https://github.com/Physical-Intelligence/openpi.git $OPENPI_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install "transformers==4.53.2"
python -m pip install -e "$REPO_ROOT" --no-deps

mkdir -p "$SITE_PACKAGES"
printf '%s\n' "/opt/venv/lib/python3.12/site-packages" > "$SITE_PACKAGES/_opt_venv.pth"

cp -r "$OPENPI_DIR/src/openpi/models_pytorch/transformers_replace/"* "$SITE_PACKAGES/transformers/"

python - <<'PY'
import inspect
import transformers
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

print("[setup] transformers", transformers.__version__)
print("[setup] transformers file", transformers.__file__)
print("[setup] GemmaRMSNorm.forward", inspect.signature(GemmaRMSNorm.forward))
PY

echo "[setup] done"
