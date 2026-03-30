#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PARENT_DIR="$(cd -- "$REPO_ROOT/.." && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$PARENT_DIR/openpi}"
OPENPI_REPO="${OPENPI_REPO:-https://github.com/Physical-Intelligence/openpi.git}"
VENV_DIR="${EVORL_VENV_DIR:-$REPO_ROOT/.venvs/pi05-openpi-ssp}"

echo "[bootstrap] repo root:  $REPO_ROOT"
echo "[bootstrap] openpi dir: $OPENPI_DIR"
echo "[bootstrap] venv dir:   $VENV_DIR"

if [ ! -d "$OPENPI_DIR/.git" ]; then
  echo "[bootstrap] cloning OpenPI into $OPENPI_DIR"
  git clone --depth=1 --recurse-submodules "$OPENPI_REPO" "$OPENPI_DIR"
else
  echo "[bootstrap] found existing OpenPI checkout at $OPENPI_DIR"
fi

OPENPI_DIR="$OPENPI_DIR" EVORL_VENV_DIR="$VENV_DIR" bash "$SCRIPT_DIR/setup_rocm_pi05.sh"

cat <<EOF

[bootstrap] next steps

1. Activate the environment:
   source "$VENV_DIR/bin/activate"

2. Authenticate Hugging Face for gated PaliGemma access:
   hf auth login

3. Optional, if you want online tracking:
   wandb login

4. Validate the core pi05 path:
   pytest -q tests/training/test_acp_pi05_prompt_pipeline.py tests/policies/pi0_pi05/test_pi05.py

5. Run a smoke workflow:
   bash scripts/experiments/pi05_acp/run_pi05_acp_smoke.sh

6. Run the full staged workflow:
   bash scripts/experiments/pi05_acp/run_pi05_acp_full_rerun.sh rerun_\$(date +%Y%m%d)

EOF
