#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/models_alehe/phi-fbsh/Evo-RL-Phi
PY=/mnt/models_alehe/phi-fbsh/.venvs/vlash-rocm-mi300x/bin/python
OUT=/mnt/models_alehe/phi-fbsh/so101_handover_hil_rl/outputs/act_awr_v1_batch1_batch2_10000steps_20260506
LOGDIR=/mnt/models_alehe/phi-fbsh/so101_handover_hil_rl/logs
DATASET=/mnt/models_alehe/phi-fbsh/so101_handover_hil_rl/datasets/fbsh96/so101_handover_hil_rlready_batch1_batch2_20260506
BASE_POLICY=/mnt/models_alehe/phi-fbsh/so101_handover_hil_rl/outputs/act_awr_v0_10000steps_20260506/checkpoints/010000/pretrained_model
RUN_ID=combined_hil_batch1_batch2_act_awr_v1_20260506
TARGETS=${DATASET}/makermods_hil/${RUN_ID}/act_awr_targets_v0.parquet

mkdir -p "$LOGDIR"
if [ -e "$OUT" ]; then
  echo "Output directory already exists: $OUT" >&2
  exit 2
fi
if [ ! -d "$BASE_POLICY" ]; then
  echo "Missing base policy: $BASE_POLICY" >&2
  exit 2
fi
if [ ! -f "$TARGETS" ]; then
  echo "Missing ACT-AWR targets: $TARGETS" >&2
  exit 2
fi

cd "$REPO"
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm-7.2.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

printf "Started ACT-AWR v1 10000-step training at %s\n" "$(date --iso-8601=seconds)"
printf "Output: %s\n" "$OUT"
printf "Dataset: %s\n" "$DATASET"
printf "Base policy: %s\n" "$BASE_POLICY"
printf "Targets: %s\n" "$TARGETS"

"$PY" -m lerobot.scripts.lerobot_train \
  --policy.path="$BASE_POLICY" \
  --dataset.repo_id=fbsh96/so101_handover_hil_rlready_batch1_batch2_20260506 \
  --dataset.root="$DATASET" \
  --output_dir="$OUT" \
  --job_name=so101_handover_act_awr_v1_batch1_batch2_10000steps \
  --batch_size=16 \
  --steps=10000 \
  --log_freq=50 \
  --save_checkpoint=true \
  --save_freq=1000 \
  --use_act_awr=true \
  --act_awr_targets_path="$TARGETS" \
  --act_awr_run_id="$RUN_ID" \
  --act_awr_weight_column=act_awr_weight \
  --act_awr_missing_weight=1.0 \
  --act_awr_normalize_batch=true \
  --act_awr_epsilon=1e-6 \
  --wandb.enable=false

mkdir -p "$OUT/selected_checkpoints"
for step in 002000 005000 010000; do
  if [ ! -d "$OUT/checkpoints/$step/pretrained_model" ]; then
    echo "Missing expected checkpoint: $OUT/checkpoints/$step/pretrained_model" >&2
    exit 2
  fi
  ln -sfn "../checkpoints/$step/pretrained_model" "$OUT/selected_checkpoints/${step}_pretrained_model"
done

printf "Finished ACT-AWR v1 10000-step training at %s\n" "$(date --iso-8601=seconds)"
printf "Selected checkpoints:\n"
find "$OUT/selected_checkpoints" -maxdepth 1 -type l -ls | sort
