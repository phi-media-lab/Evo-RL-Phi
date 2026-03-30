#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
VENV="${EVORL_VENV_BIN:-$REPO_ROOT/.venvs/pi05-openpi-ssp/bin}"
DATASET_REPO="maxbeau/XLeRobot"
DATASET_ROOT="${LEROBOT_DATASET_ROOT:-/root/.cache/huggingface/lerobot/maxbeau/XLeRobot}"
RUN_SUFFIX="${1:-rerun_20260329}"

export HF_HUB_ENABLE_HF_TRANSFER=1

run_stage() {
  local stage="$1"
  local value_steps="$2"
  local value_save_freq="$3"
  local infer_batch_size="$4"
  local infer_n_step="$5"
  local policy_steps="$6"
  local policy_save_freq="$7"
  local policy_save_checkpoint="$8"

  local value_run_dir="$REPO_ROOT/outputs/value_train/pi05_acp_${stage}_${RUN_SUFFIX}"
  local value_infer_dir="$REPO_ROOT/outputs/value_infer/pi05_acp_${stage}_${RUN_SUFFIX}"
  local policy_run_dir="$REPO_ROOT/outputs/train/pi05_acp_policy_${stage}_${RUN_SUFFIX}"

  local value_field="complementary_info.value_${stage}_${RUN_SUFFIX}"
  local adv_field="complementary_info.advantage_${stage}_${RUN_SUFFIX}"
  local ind_field="complementary_info.acp_indicator_${stage}_${RUN_SUFFIX}"

  cd "$REPO_ROOT"

  "$VENV/lerobot-value-train" \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.video_backend=pyav \
    --value.type=pistar06 \
    --value.device=cuda \
    --value.dtype=float32 \
    --value.normalization_mapping='{"ACTION":"IDENTITY","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
    --batch_size=1 \
    --num_workers=0 \
    --steps="$value_steps" \
    --save_checkpoint=true \
    --save_freq="$value_save_freq" \
    --wandb.enable=false \
    --output_dir="$value_run_dir" \
    --job_name="pi05_acp_${stage}_${RUN_SUFFIX}_value"

  "$VENV/lerobot-value-infer" \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.video_backend=pyav \
    --dataset.default_success=failure \
    --inference.checkpoint_path="$value_run_dir" \
    --runtime.device=cuda \
    --runtime.batch_size="$infer_batch_size" \
    --runtime.num_workers=0 \
    --acp.enable=true \
    --acp.n_step="$infer_n_step" \
    --acp.positive_ratio=0.3 \
    --acp.value_field="$value_field" \
    --acp.advantage_field="$adv_field" \
    --acp.indicator_field="$ind_field" \
    --output_dir="$value_infer_dir" \
    --job_name="pi05_acp_${stage}_${RUN_SUFFIX}_infer"

  "$VENV/lerobot-train" \
    --dataset.repo_id="$DATASET_REPO" \
    --dataset.root="$DATASET_ROOT" \
    --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.device=cuda \
    --policy.dtype=float32 \
    --policy.push_to_hub=false \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
    --batch_size=1 \
    --num_workers=0 \
    --steps="$policy_steps" \
    --save_checkpoint="$policy_save_checkpoint" \
    --save_freq="$policy_save_freq" \
    --wandb.enable=false \
    --acp.enable=true \
    --acp.indicator_field="$ind_field" \
    --acp.indicator_dropout_prob=0.3 \
    --output_dir="$policy_run_dir" \
    --job_name="pi05_acp_policy_${stage}_${RUN_SUFFIX}"
}

run_stage smoke 1 1 1 5 1 1 false
run_stage pilot 5 5 4 10 5 5 true
run_stage stage1 20 10 4 10 20 10 true
run_stage stage2 100 50 4 10 100 50 true
run_stage stage3 500 250 4 10 500 250 true
