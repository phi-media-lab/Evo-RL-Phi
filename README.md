<h1 align="center">Evo-RL</h1>

<p align="center">
  <a href="https://github.com/huggingface/lerobot"><img alt="lerobot version" src="https://img.shields.io/badge/LeRobot-0.4.4-f59e0b"/></a>
  <a href="./LICENSE"><img alt="license" src="https://img.shields.io/badge/License-Apache--2.0-ef4444"/></a>
</p>

<p align="center"><strong>Real-world RL workflows on top of LeRobot, with a validated ROCm + MI300X + pi05 ACP path.</strong></p>

## What This Repo Is

`Evo-RL` is a LeRobot-based codebase for real-world robot learning. The repository currently focuses on:

- offline RL and iterative improvement workflows for real robots
- SO101 and AgileX PiPER / PiPER-X hardware support
- value training, value inference, ACP-tagged policy training, and rollout collection
- a single-repo ROCm deployment path for `pi05` on MI300X

This branch is organized so a new MI300X cloud machine can start from this repository alone, instead of depending on an external hackathon repo or local path conventions.

## What Is Validated On This Branch

The `pi05-rocm-acp` branch has already been validated for the ROCm + `pi05` ACP workflow on MI300X-class hardware:

- fresh clone of this repo
- bootstrap of a local `.venvs/pi05-openpi-ssp`
- OpenPI-patched `transformers==4.53.2`
- `pytest -q tests/training/test_acp_pi05_prompt_pipeline.py tests/policies/pi0_pi05/test_pi05.py`
- `pi05` ACP smoke workflow:
  - `value train`
  - `value infer`
  - `policy train`
- staged reruns up to `stage3` / `500` training steps

Operational constraints for the validated path:

- ROCm-capable MI300X machine
- Hugging Face access to `google/paligemma-3b-pt-224`
- `pyav` as the video backend
- dataset-compatible cache or first-download access

## Start Here

Choose one path.

| Goal | Entry |
| --- | --- |
| Bring up a fresh cloud MI300X instance | [`scripts/setup/bootstrap_runpod_mi300x.sh`](./scripts/setup/bootstrap_runpod_mi300x.sh) |
| Install into an existing ROCm machine | [`scripts/setup/setup_rocm_pi05.sh`](./scripts/setup/setup_rocm_pi05.sh) |
| Read the MI300X workflow overview | [`docs/source/pi05_acp_mi300x.mdx`](./docs/source/pi05_acp_mi300x.mdx) |
| Follow the shortest cloud setup path | [`docs/source/pi05_acp_runpod_quickstart.mdx`](./docs/source/pi05_acp_runpod_quickstart.mdx) |
| Run the minimal end-to-end check | [`scripts/experiments/pi05_acp/run_pi05_acp_smoke.sh`](./scripts/experiments/pi05_acp/run_pi05_acp_smoke.sh) |
| Re-run the full staged validation | [`scripts/experiments/pi05_acp/run_pi05_acp_full_rerun.sh`](./scripts/experiments/pi05_acp/run_pi05_acp_full_rerun.sh) |

## Fastest Path

Fresh MI300X cloud instance:

```bash
git clone https://github.com/phi-media-lab/Evo-RL-Phi.git
cd Evo-RL-Phi
bash scripts/setup/bootstrap_runpod_mi300x.sh
source .venvs/pi05-openpi-ssp/bin/activate
hf auth login
pytest -q tests/training/test_acp_pi05_prompt_pipeline.py tests/policies/pi0_pi05/test_pi05.py
bash scripts/experiments/pi05_acp/run_pi05_acp_smoke.sh
```

If you want the staged workflow after smoke:

```bash
bash scripts/experiments/pi05_acp/run_pi05_acp_full_rerun.sh rerun_YYYYMMDD
```

## Repo Map

- `scripts/setup/`
  - ROCm environment bootstrap and OpenPI patching
- `scripts/experiments/pi05_acp/`
  - smoke, pilot, stage1, stage2, stage3, and full rerun scripts
- `docs/source/pi05_acp_mi300x.mdx`
  - single-repo workflow overview
- `docs/source/pi05_acp_runpod_quickstart.mdx`
  - shortest cloud setup path
- `docs/source/pi05_acp/`
  - execution plan, runbook, experiment report
- `src/lerobot/`
  - core training, policy, value, robot, and environment code

## Table of Contents

| Getting Started | Training Pipeline | Project Info |
| --- | --- | --- |
| [Quick Start](#quick-start) | [4) Value Function Training](#value-function-training) | [Model & Dataset](#model--dataset) |
| [1) Installation](#installation) | [5) Value Inference](#value-inference) | [Community Channels](#community-channels) |
| [2) Hardware Setup](#hardware-setup) | [6) Policy Training](#policy-training) | [Affiliations](#affiliations) |
| [3) Data Collection](#data-collection) | [7) Closed-loop Rollout and Next Round](#closed-loop-rollout-and-next-round) | [Citation](#citation) / [License](#license) |

<a id="quick-start"></a>

## ⚡ Quick Start

This repository is built on top of LeRobot and extends it for real-world RL workflows.

### MI300X / `pi05` ACP

For the ROCm + MI300X + `pi05` ACP workflow, this repository is the only required entrypoint.

Primary docs and scripts:

- `docs/source/pi05_acp_mi300x.mdx`
- `docs/source/pi05_acp_runpod_quickstart.mdx`
- `scripts/setup/bootstrap_runpod_mi300x.sh`
- `scripts/setup/setup_rocm_pi05.sh`
- `scripts/experiments/pi05_acp/`

Validation command:

```bash
pytest -q tests/training/test_acp_pi05_prompt_pipeline.py tests/policies/pi0_pi05/test_pi05.py
```

Minimal end-to-end run:

```bash
bash scripts/experiments/pi05_acp/run_pi05_acp_smoke.sh
```

Full staged rerun:

```bash
bash scripts/experiments/pi05_acp/run_pi05_acp_full_rerun.sh rerun_YYYYMMDD
```

Fresh MI300X cloud bootstrap:

```bash
bash scripts/setup/bootstrap_runpod_mi300x.sh
```

<a id="installation"></a>

### 1) Installation

For the validated MI300X path, prefer the repository-managed scripts instead of manual setup:

```bash
git clone https://github.com/phi-media-lab/Evo-RL-Phi.git
cd Evo-RL-Phi
bash scripts/setup/bootstrap_runpod_mi300x.sh
```

For a generic development install:

```bash
git clone https://github.com/phi-media-lab/Evo-RL-Phi.git
cd Evo-RL-Phi
conda create -y -n evo-rl python=3.10
conda activate evo-rl
pip install -e .
```

For setup details and platform-specific dependencies, follow the official [LeRobot configuration guide](https://huggingface.co/docs/lerobot/installation).

<a id="hardware-setup"></a>

### 2) Hardware Setup

#### SO Series (SO100/SO101)

For SO-series setup, please follow the [official tutorial](https://wiki.seeedstudio.com/cn/lerobot_so100m/) in detail and complete all installation and configuration steps there before continuing.
The examples below use **SO101** as the reference configuration.

#### Device path recommendation

Recommended path strategy:

- **Robot serial:** use `/dev/serial/by-id/` (stable across reboots).
- **Cameras:** prefer `/dev/v4l/by-id/`; if IDs are not unique, use `/dev/v4l/by-path/`.
- In examples below: robot ports use `by-id`, camera paths use `by-path`.

You can inspect available stable paths with:

```bash
ls -l /dev/serial/by-id/
ls -l /dev/v4l/by-id/
ls -l /dev/v4l/by-path/
```

For single-arm users, no major changes are required. After setup, run the command below to verify your system is ready for the next stage:

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/<SO101_FOLLOWER_PORT> \
  --robot.id=my_so101_follower \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/<SO101_LEADER_PORT> \
  --teleop.id=my_so101_leader
```

For dual-arm users, we recommend mirroring the mechanical parts corresponding to servos 4/5/6 on the left leader and left follower arms, which usually provides a more natural bimanual operation feel.

Before running the dual-arm command, make sure calibration files exist under `~/.cache/huggingface/lerobot/calibration/` like:

```text
calibration/
├── robots
│   └── so_follower
│       ├── bi_so101_follower_left.json
│       └── bi_so101_follower_right.json
└── teleoperators
    └── so_leader
        ├── bi_so101_leader_left.json
        └── bi_so101_leader_right.json
```

This layout is slightly different from single-arm setup.

Then run this command to verify dual-arm setup:

```bash
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/<LEFT_FOLLOWER_PORT> \
  --robot.right_arm_config.port=/dev/serial/by-id/<RIGHT_FOLLOWER_PORT> \
  --robot.id=bi_so101_follower \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/<LEFT_LEADER_PORT> \
  --teleop.right_arm_config.port=/dev/serial/by-id/<RIGHT_LEADER_PORT> \
  --teleop.id=bi_so101_leader
```

#### Camera configuration

Before data collection, validate camera mapping first.

Check whether each camera supports your target setting (for example, `640x480 @ 30`):

```bash
v4l2-ctl -d /dev/v4l/by-path/<CAM_PATH> --list-formats-ext
```

Single-arm camera check (example):

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/<SO101_FOLLOWER_PORT> \
  --robot.id=my_so101_follower \
  --robot.cameras='{ front: {type: opencv, index_or_path: "/dev/v4l/by-path/<FRONT_CAM>", width: 640, height: 480, fps: 30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/<SO101_LEADER_PORT> \
  --teleop.id=my_so101_leader \
  --display_data=true
```

Dual-arm camera check (example):

```bash
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/<LEFT_FOLLOWER_PORT> \
  --robot.right_arm_config.port=/dev/serial/by-id/<RIGHT_FOLLOWER_PORT> \
  --robot.id=my_bi_so101_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<LEFT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<RIGHT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: "/dev/v4l/by-path/<FRONT_CAM_PATH>", width: 640, height: 480, fps: 30}}' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/<LEFT_LEADER_PORT> \
  --teleop.right_arm_config.port=/dev/serial/by-id/<RIGHT_LEADER_PORT> \
  --teleop.id=my_bi_so101_leader \
  --display_data=true
```

For dual-arm camera mapping, it is fine to attach `front` under either the left-arm or right-arm camera config. If you use more camera views, place them under either the left or right arm camera config as well.

If needed, you can also use temporary device paths (for example `/dev/ttyACM*` and `/dev/video*`) during initial debugging.

<a id="agilex-piper-setup"></a>

#### AgileX (PiPER/PiPER-X)

PiPER arms in master/teaching mode cannot receive external control commands, so all arms must be configured to follower/motion-output mode (0xFC), and firmware must be version 1.8.5 or above.

For PiPER-series robots, make sure Git LFS assets are pulled before running teleoperation:

```bash
git lfs pull --include="src/lerobot/assets/piper_description/**,src/lerobot/assets/piper_x_description/**" --exclude="*"
git lfs checkout src/lerobot/assets/piper_description src/lerobot/assets/piper_x_description
```

For PiPER setup, PiPER uses CAN interfaces instead of serial ports.
So first run `lerobot-setup-can` to confirm CAN interfaces are available:

```bash
lerobot-setup-can --mode=setup --interfaces=<LEFT_FOLLOWER_CAN_PORT>,<LEFT_LEADER_CAN_PORT>,<RIGHT_FOLLOWER_CAN_PORT>,<RIGHT_LEADER_CAN_PORT>
```

For single-arm users, run the command below to verify the system is ready:

```bash
lerobot-teleoperate \
  --robot.type=piperx_follower \
  --robot.port=<FOLLOWER_CAN_PORT> \
  --robot.id=my_piperx_follower \
  --robot.require_calibration=false \
  --teleop.type=piperx_leader \
  --teleop.port=<LEADER_CAN_PORT> \
  --teleop.id=my_piperx_leader \
  --teleop.require_calibration=false
```

For bimanual users, run this command to verify dual-arm teleoperation:

```bash
lerobot-teleoperate \
  --robot.type=bi_piperx_follower \
  --robot.id=my_bi_piperx_follower \
  --robot.left_arm_config.port=<LEFT_FOLLOWER_CAN_PORT> \
  --robot.right_arm_config.port=<RIGHT_FOLLOWER_CAN_PORT> \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --teleop.type=bi_piperx_leader \
  --teleop.id=my_bi_piperx_leader \
  --teleop.left_arm_config.port=<LEFT_LEADER_CAN_PORT> \
  --teleop.right_arm_config.port=<RIGHT_LEADER_CAN_PORT> \
  --teleop.left_arm_config.require_calibration=false \
  --teleop.right_arm_config.require_calibration=false
```

For PiPER (non-X), replace `bi_piperx_follower`/`bi_piperx_leader` with `bi_piper_follower`/`bi_piper_leader`.

<a id="data-collection"></a>

### 3) Data Collection

Collect rollout data with `lerobot-human-inloop-record`.

#### SO Series (SO100/SO101)

Bimanual template:

```bash
lerobot-human-inloop-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/<LEFT_FOLLOWER_PORT> \
  --robot.right_arm_config.port=/dev/serial/by-id/<RIGHT_FOLLOWER_PORT> \
  --robot.id=my_bi_so101_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<LEFT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<RIGHT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: intelrealsense, serial_number_or_name: "<REALSENSE_SN>", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/<LEFT_LEADER_PORT> \
  --teleop.right_arm_config.port=/dev/serial/by-id/<RIGHT_LEADER_PORT> \
  --teleop.id=my_bi_so101_leader \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --dataset.single_task="<YOUR_TASK_DESCRIPTION>" \
  --dataset.num_episodes=<NUM_EPISODES> \
  --dataset.episode_time_s=<EPISODE_SECONDS> \
  --dataset.reset_time_s=<RESET_SECONDS> \
  --dataset.push_to_hub=true \
  --display_data=true
```

Recommendation: use **`fourcc: "MJPG"`** for OpenCV and **`warmup_s`** for RealSense. In this example `front` uses RealSense, but you can switch it to OpenCV with the same structure.

#### AgileX (PiPER/PiPER-X)

Bimanual template (left/right, PiPER-X example):

```bash
lerobot-human-inloop-record \
  --robot.type=bi_piperx_follower \
  --robot.id=my_bi_piperx_follower \
  --robot.left_arm_config.port=<LEFT_FOLLOWER_CAN_PORT> \
  --robot.right_arm_config.port=<RIGHT_FOLLOWER_CAN_PORT> \
  --robot.left_arm_config.require_calibration=false \
  --robot.right_arm_config.require_calibration=false \
  --teleop.type=bi_piperx_leader \
  --teleop.id=my_bi_piperx_leader \
  --teleop.left_arm_config.port=<LEFT_LEADER_CAN_PORT> \
  --teleop.right_arm_config.port=<RIGHT_LEADER_CAN_PORT> \
  --teleop.left_arm_config.require_calibration=false \
  --teleop.right_arm_config.require_calibration=false \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --dataset.single_task="<YOUR_TASK_DESCRIPTION>" \
  --dataset.num_episodes=<NUM_EPISODES> \
  --dataset.episode_time_s=<EPISODE_SECONDS> \
  --dataset.reset_time_s=<RESET_SECONDS> \
  --dataset.push_to_hub=true \
  --display_data=true
```

Hotkeys:

- `i`: toggle intervention mode (policy <-> teleop takeover)
- `s`: mark success and end current episode
- `f`: mark failure and end current episode
- `Right Arrow`: end the current loop early
- `Left Arrow`: end early and re-record the current episode
- `Esc`: stop the recording session

Quick quality check:

```bash
lerobot-dataset-report --dataset <HF_USERNAME_OR_ORG>/<DATASET_NAME>
```

This prints: dataset meta, totals, episode-length stats/histogram, success/intervention metrics, task list, and full feature schema.

<a id="value-function-training"></a>

### 4) Value Function Training

Train the value function on the current dataset. Current default: [Pi\*0.6](https://www.pi.website/blog/pistar06) (`--value.type=pistar06`).

Single-GPU template:

```bash
lerobot-value-train \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --value.type=pistar06 \
  --value.dtype=bfloat16 \
  --value.push_to_hub=true \
  --value.repo_id=<HF_USERNAME_OR_ORG>/<VALUE_MODEL_REPO> \
  --batch_size=64 \
  --output_dir=outputs/value_train/<RUN_NAME> \
  --job_name=<RUN_NAME> \
  --wandb.enable=true
```

Multi-GPU template:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID_LIST> accelerate launch \
  --multi_gpu \
  --num_processes=<NUM_GPUS> \
  --mixed_precision=bf16 \
  $(which lerobot-value-train) \
  --batch_size=32/<NUM_GPUS> \
  <VALUE_TRAIN_ARGS>
```

To plug in a different value function, minimal path in this repo:

- Add `src/lerobot/values/<your_value>/configuration_<your_value>.py` with `@PreTrainedConfig.register_subclass("<your_value>")`.
- Add `src/lerobot/values/<your_value>/modeling_<your_value>.py` with `<YourValue>Policy(PreTrainedPolicy)` (implement at least `forward`, `predict_value`, and `build_training_raw_batch_hook` for `lerobot-value-train`).
- Add `src/lerobot/values/<your_value>/processor_<your_value>.py` with `make_<your_value>_pre_post_processors(...)`.
- Remove/replace the current `pistar06`-only type checks in `src/lerobot/configs/value_train.py` and `src/lerobot/scripts/lerobot_value_infer.py`.

<a id="value-inference"></a>

### 5) Value Inference

Infer value signals and write value/advantage/indicator back to the dataset:

- `value`: estimated return-to-go of the current frame.
- `advantage`: relative improvement signal (higher means better-than-baseline trajectory quality).
- `indicator`: binarized training tag derived from advantage.

Single-GPU template:

```bash
lerobot-value-infer \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --inference.checkpoint_path=outputs/value_train/<RUN_NAME> \
  --runtime.device=cuda \
  --runtime.batch_size=64 \
  --acp.enable=true \
  --acp.n_step=50 \
  --acp.positive_ratio=0.3 \
  --acp.value_field=complementary_info.value_<TAG> \
  --acp.advantage_field=complementary_info.advantage_<TAG> \
  --acp.indicator_field=complementary_info.acp_indicator_<TAG> \
  --output_dir=outputs/value_infer/<RUN_NAME> \
  --job_name=<RUN_NAME>.infer
```

Multi-GPU template:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID_LIST> accelerate launch \
  --multi_gpu \
  --num_processes=<NUM_GPUS> \
  --mixed_precision=bf16 \
  $(which lerobot-value-infer) \
  <VALUE_INFER_ARGS>
```

Parameter notes:

```bash
--acp.n_step: n-step advantage horizon.
--acp.positive_ratio: positive label ratio after advantage binarization (e.g., 0.3 = top 30% per task).
```

Expected new columns:

```bash
complementary_info.value_<TAG>
complementary_info.advantage_<TAG>
complementary_info.acp_indicator_<TAG>
```

These columns are written back to the original dataset specified by `--dataset.repo_id`.

<a id="policy-training"></a>

### 6) Policy Training

Train the policy with advantage-conditioned tags.
**Policy requirement:** it must support **text/task input**, because Advantage-Conditioned tags are injected into **task text**.

Single-GPU template:

```bash
lerobot-train \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME> \
  --policy.type=<POLICY_TYPE> \
  --policy.pretrained_path=<POLICY_PRETRAINED_PATH> \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --batch_size=32 \
  --steps=30000 \
  --acp.enable=true \
  --acp.indicator_field=complementary_info.acp_indicator_<TAG> \
  --acp.indicator_dropout_prob=0.3 \
  --output_dir=outputs/train/<RUN_NAME> \
  --job_name=<RUN_NAME> \
  --wandb.enable=true \
  --policy.push_to_hub=true \
  --policy.repo_id=<HF_USERNAME_OR_ORG>/<POLICY_REPO>
```

`--acp.indicator_dropout_prob` controls tag drop rate in task text; `0.3` helps learn both tagged and untagged conditions.

Important checks:

- `--acp.indicator_field` must exist in the dataset and be **binary (`0/1`)**.

Multi-GPU template:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID_LIST> accelerate launch \
  --multi_gpu \
  --num_processes=<NUM_GPUS> \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --batch_size=32/<NUM_GPUS> \
  <POLICY_TRAIN_ARGS>
```

<a id="closed-loop-rollout-and-next-round"></a>

### 7) Closed-loop Rollout and Next Round

Deploy the trained policy in human-in-loop mode and collect the next dataset round:

```bash
lerobot-human-inloop-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/serial/by-id/<LEFT_FOLLOWER_PORT> \
  --robot.right_arm_config.port=/dev/serial/by-id/<RIGHT_FOLLOWER_PORT> \
  --robot.id=my_bi_so101_follower \
  --robot.left_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<LEFT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --robot.right_arm_config.cameras='{ wrist: {type: opencv, index_or_path: "/dev/v4l/by-path/<RIGHT_WRIST_CAM_PATH>", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, front: {type: intelrealsense, serial_number_or_name: "<REALSENSE_SN>", width: 640, height: 480, fps: 30, warmup_s: 2}}' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/serial/by-id/<LEFT_LEADER_PORT> \
  --teleop.right_arm_config.port=/dev/serial/by-id/<RIGHT_LEADER_PORT> \
  --teleop.id=my_bi_so101_leader \
  --dataset.repo_id=<HF_USERNAME_OR_ORG>/<DATASET_NAME_NEXT_ROUND> \
  --dataset.single_task="<YOUR_TASK_DESCRIPTION>" \
  --dataset.num_episodes=<NUM_EPISODES> \
  --dataset.episode_time_s=<EPISODE_SECONDS> \
  --dataset.reset_time_s=<RESET_SECONDS> \
  --dataset.push_to_hub=true \
  --display_data=true \
  --policy.path=<POLICY_CHECKPOINT_OR_HUB_ID> \
  --resume=true
```

Dataset continuation options:

- **Append in place:** keep `--resume=true` and continue recording into the same dataset.
- **Merge multiple rounds:** use the official dataset editor to merge separate datasets.

```bash
lerobot-edit-dataset \
  --repo_id=<HF_USERNAME_OR_ORG>/<MERGED_DATASET_NAME> \
  --operation.type=merge \
  --operation.repo_ids="['<HF_USERNAME_OR_ORG>/<DATASET_ROUND_1>','<HF_USERNAME_OR_ORG>/<DATASET_ROUND_2>']"
```

Additional data attributes vs default `lerobot-record` behavior:

- `complementary_info.policy_action`: policy output action at each step.
- `complementary_info.is_intervention`: whether current step is in intervention.
- `complementary_info.state`: intervention state-machine state.
- `complementary_info.collector_policy_id`: step-level action source ID (`human` or policy ID).
- Episode metadata `episode_success`: success/failure label saved per episode.

Iterative training loop (abstract):

```text
[Multi-task demonstration data pool]
        |
        v
[Offline RL pretraining for a vision-language-action policy]
        |
        v
[Task-specific initialization / fine-tuning from demonstrations]
        |
        v
|---- Iteration k = 1..K -------------------------------------|
| 1) Deploy current policy π_k and collect new rollout data   |
| 2) Merge into data pool: D <- D U new_data                  |
| 3) Train value function on D                                |
| 4) Infer advantage and binarize into indicator tags         |
| 5) Train advantage-conditioned policy to get π_{k+1}        |
|-------------------------------------------------------------|
        |
        v
[Stronger policy with improved success rate and throughput]
```

## Model & Dataset

- Hugging Face model release: coming soon
- Hugging Face dataset release: coming soon
- Once published, this section will pin canonical repos and exact version tags.

## Community Channels

- WeChat official post: [Coming Soon](https://evorl.example.com/wechat-post)
- Documentation: [`docs/README.md`](./docs/README.md)
- GitHub Issues: [Create an issue](https://github.com/phi-media-lab/Evo-RL-Phi/issues)
- Email: business@evomind-tech.com
- WeChat group QR code:

<p align="center">
  <img alt="EvoMind WeChat QR" src="./website/assets/images/rlgroup.jpg" width="220"/>
</p>

## Affiliations

<p align="center">
  <img alt="SJTU community visual" src="./website/assets/images/sjtu.png" height="68"/>
  <img alt="EvoMind" src="./website/assets/images/evomind1.png" height="60"/>
</p>

## Citation

```bibtex
@misc{evorl2026,
  title        = {Evo-RL: Towards Iterative Policy Improvement in Real-World Offline RL},
  author       = {Evo-RL Contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/phi-media-lab/Evo-RL-Phi}}
}
```

## License

Apache-2.0. See [LICENSE](./LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/image?repos=phi-media-lab/Evo-RL-Phi&type=date&legend=top-left)](https://www.star-history.com/?repos=phi-media-lab%2FEvo-RL-Phi&type=date&legend=top-left)
