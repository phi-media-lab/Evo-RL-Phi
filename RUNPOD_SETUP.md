# Runpod Setup

这份文档用于把 Evo-RL 的云端开发环境部署到 Runpod 服务器，并为后续的 `ingestion / materializer / training / artifact build` 做准备。

## 1. 目标

Runpod 侧先达到这几个状态：

- 仓库可以正常 clone / pull
- Python 3.10 环境可用
- `pip install -e .` 成功
- `torch.cuda.is_available()` 为 `True`
- Hugging Face / Wandb / cache 目录约定清晰
- 持久化目录固定
- 断开终端后任务仍能继续运行

如果这些还没稳定，不建议继续推进云端模块实现。

## 2. 推荐目录布局

假设 Runpod 的持久卷挂载到 `/workspace`。

建议使用：

```text
/workspace/
  Evo-RL/                  # 仓库
  cache/
    huggingface/
    pip/
    uv/
  data/
    uploads/
    ingestion/
    materialized/
  outputs/
    train/
    eval/
  artifacts/
  logs/
```

先创建这些目录：

```bash
mkdir -p /workspace/cache/huggingface
mkdir -p /workspace/cache/pip
mkdir -p /workspace/cache/uv
mkdir -p /workspace/data/uploads
mkdir -p /workspace/data/ingestion
mkdir -p /workspace/data/materialized
mkdir -p /workspace/outputs/train
mkdir -p /workspace/outputs/eval
mkdir -p /workspace/artifacts
mkdir -p /workspace/logs
```

## 3. 系统准备

如果 Runpod 镜像已经带 `git`、`wget`、`tmux`、`conda`，可以跳过对应步骤。

最低要求：

- `git`
- `wget` 或 `curl`
- `tmux`
- `conda` 或 `micromamba`
- NVIDIA 驱动已正常工作

快速检查：

```bash
nvidia-smi
git --version
tmux -V
conda --version
```

## 4. 克隆仓库

建议直接拉你的 fork：

```bash
cd /workspace
git clone https://github.com/phi-media-lab/Evo-RL-Phi.git Evo-RL
cd /workspace/Evo-RL
git remote add upstream https://github.com/MINT-SJTU/Evo-RL.git
git remote -v
```

如果后续要接着当前开发分支：

```bash
git fetch fork || true
git checkout edge-control-plane-foundation
```

如果远端还没配置 `fork` remote，也可以直接：

```bash
git fetch origin edge-control-plane-foundation
git checkout edge-control-plane-foundation
```

## 5. Python 环境

推荐直接建立一个独立环境：

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
cd /workspace/Evo-RL
pip install --upgrade pip
pip install -e .
```

如果要让 pip cache 走持久卷：

```bash
export PIP_CACHE_DIR=/workspace/cache/pip
```

## 6. Runpod 环境变量

建议把下面这些变量写进一个 shell 文件，例如 `/workspace/runpod_env.sh`：

```bash
export HF_HOME=/workspace/cache/huggingface
export PIP_CACHE_DIR=/workspace/cache/pip
export WANDB_DIR=/workspace/logs/wandb
export TOKENIZERS_PARALLELISM=false
```

如果你要从 Hugging Face 拉模型/数据：

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
```

如果你要用 Wandb：

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
wandb login --relogin "$WANDB_API_KEY"
```

生效方式：

```bash
source /workspace/runpod_env.sh
```

## 7. 安装验证

环境装好后，至少跑这些检查：

```bash
cd /workspace/Evo-RL
conda activate lerobot
python -c "import torch, lerobot; print(torch.__version__); print(torch.cuda.is_available())"
python -m compileall src/lerobot/edge src/lerobot/control_plane src/lerobot/cloud
```

再跑当前 edge/control-plane 回归：

```bash
pytest -q tests/edge/test_control_plane_rollout_report.py \
  tests/edge/test_rollout_status.py \
  tests/edge/test_incident_aggregator.py \
  tests/edge/test_deployment_loop.py \
  tests/edge/test_model_manager_sync.py \
  tests/edge/test_edge_run_local.py \
  tests/edge/test_release_registry.py \
  tests/edge/test_edge_runner.py \
  tests/edge/test_edge_uploader.py
```

## 8. 持续会话

Runpod 上不要直接依赖单个 SSH 会话。

建议使用 `tmux`：

```bash
tmux new -s evorl
```

常用操作：

```bash
tmux attach -t evorl
tmux ls
```

统一 cloud stack 可以直接挂在 `tmux` 里运行，例如：

```bash
cd /workspace/Evo-RL
source /workspace/runpod_env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot

python -m lerobot.scripts.cloud_stack \
  --host 127.0.0.1 \
  --ingestion-port 8000 \
  --materializer-port 8001 \
  --status-port 8002 \
  --ingestion-root /workspace/data/ingestion \
  --materialized-root /workspace/data/materialized \
  --dataset-root /workspace/data/dataset \
  --train-output-root /workspace/outputs/train \
  --artifact-output-root /workspace/artifacts \
  --registry-root /workspace/data/registry \
  --state-root /workspace/data/controller_state \
  --runtime-root /workspace/logs/cloud_stack \
  --incident-root /workspace/data/incidents \
  --report-root /workspace/data/reports \
  --channel staging \
  --artifact-prefix artifact-cloud-stack \
  --robot-type mock_robot \
  --camera-layout single_arm_mock
```

启动后重点看这些文件：

- `/workspace/logs/cloud_stack/daemon.log`
- `/workspace/logs/cloud_stack/history.jsonl`
- `/workspace/logs/cloud_stack/latest.json`
- `/workspace/logs/cloud_stack/metrics.json`
- `/workspace/logs/cloud_stack/cloud_stack_status.json`

HTTP 探活和状态：

- `http://127.0.0.1:8002/healthz`
- `http://127.0.0.1:8002/status`

如果你不想手敲长命令，仓库里已经带了 `tmux` 包装脚本：

```bash
cd /workspace/Evo-RL
chmod +x scripts/cloud_stack_tmux.sh
SESSION_NAME=evorl-cloud-stack scripts/cloud_stack_tmux.sh start
SESSION_NAME=evorl-cloud-stack scripts/cloud_stack_tmux.sh status
SESSION_NAME=evorl-cloud-stack scripts/cloud_stack_tmux.sh logs
```

停止方式：

```bash
SESSION_NAME=evorl-cloud-stack scripts/cloud_stack_tmux.sh stop
```

如果你所在环境支持 `systemd`，仓库也附带了 unit 模板：

```bash
sudo cp scripts/cloud_stack.service /etc/systemd/system/evorl-cloud-stack.service
sudo systemctl daemon-reload
sudo systemctl enable --now evorl-cloud-stack.service
sudo systemctl status evorl-cloud-stack.service
```

## 9. 当前阶段推荐先做什么

Runpod 环境就绪后，优先顺序建议是：

1. 跑通当前测试，确认基础环境没问题
2. 明确 `/workspace/data/ingestion` 和 `/workspace/data/materialized` 的真实路径
3. 开始实现真实 `ingestion service`
4. 再实现 `dataset materializer`
5. 最后接训练和 artifact builder

## 10. 一键脚本

仓库里已经附带一个 bootstrap 脚本：

```bash
bash scripts/runpod_bootstrap.sh
```

这个脚本会：

- 创建推荐目录
- 创建 `lerobot` conda 环境
- 安装项目
- 生成 `/workspace/runpod_env.sh`
- 做基础导入检查

## 11. 常见问题

### `torch.cuda.is_available()` 是 `False`

先看：

```bash
nvidia-smi
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
```

如果驱动正常但 torch 不认 CUDA，通常是镜像和 torch 版本不匹配。

### `pip install -e .` 很慢

确认：

```bash
export PIP_CACHE_DIR=/workspace/cache/pip
```

### 会话断开后任务没了

说明没有放进 `tmux`。

### 路径混乱

不要把 repo、cache、outputs、artifacts 混在一起。按本文档固定目录。
