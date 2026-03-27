# Evo-RL

[English README](./README.md)

Evo-RL 是一个基于 LeRobot 的真实机器人强化学习代码库，当前重点是构建一条务实的云边协同链路：

- 在云端训练和评测策略
- 将模型打包成可部署 artifact
- 在边缘机器人本地执行策略
- 稳定落盘 episode 数据
- 将结构化数据回传到控制面
- 基于 channel 管理 rollout、rollback 和 incident

这个仓库现在已经不只是“架构设计”。它已经包含一套最小可运行骨架，覆盖：

- 本地 edge 执行
- episode spool 和 upload
- HTTP ingestion 和 materializer
- 训练 smoke 与 artifact build
- artifact 与 channel 化发布
- episode boundary 模型切换
- `model_crash` 回滚
- incident 聚合
- control-plane rollout report
- 常驻 cloud stack 编排

## 当前状态

当前已实现：

- edge runtime contract、episode schema、artifact manifest、release schema
- local runtime、watchdog、runner、recorder、spool、uploader
- 文件系统版和 HTTP 版 ingestion
- 文件系统版和 HTTP 版 materializer
- 导出本地 `LeRobotDataset`
- 一步训练 smoke 和 artifact builder
- release registry 与 device/channel mapping
- 维护 active / pending / previous-active 的 model manager
- episode boundary deployment loop
- `model_crash` 触发 rollback
- edge incident sink
- control-plane incident aggregation
- rollout status snapshot builder 与 report script
- auto-release daemon 与统一 `cloud_stack`

当前未实现：

- 生产级鉴权和存储后端
- 长时间训练 job 与真实评测 gate
- 生产级审批和 promote 流程
- 远程健康上报与生产审计链路

详细实施计划和阶段状态见 [DRAFT.md](./DRAFT.md)。

## 仓库结构

关键模块：

- [src/lerobot/edge](./src/lerobot/edge)
  本地 runtime、deployment loop、recorder、spool、uploader、watchdog、incidents。
- [src/lerobot/control_plane](./src/lerobot/control_plane)
  artifact schema、registry、incident aggregation、rollout status reporting。
- [src/lerobot/cloud](./src/lerobot/cloud)
  ingestion、materializer、artifact build 和云端辅助逻辑。
- [src/lerobot/scripts/edge_run_local.py](./src/lerobot/scripts/edge_run_local.py)
  边缘执行入口。
- [src/lerobot/scripts/cloud_stack.py](./src/lerobot/scripts/cloud_stack.py)
  统一云端栈入口。
- [src/lerobot/scripts/control_plane_rollout_report.py](./src/lerobot/scripts/control_plane_rollout_report.py)
  控制面 report 入口。

## 环境准备

推荐本地环境：

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
pip install -e .
```

验证安装：

```bash
python -c "import lerobot; print(lerobot.__version__)"
```

本仓库的 agent 约定见 [AGENTS.md](./AGENTS.md)。

## 主要入口

### Edge 执行入口

查看本地 edge loop 参数：

```bash
python -m lerobot.scripts.edge_run_local --help
```

当前支持：

- dry-run 执行
- 本地 spool 与可选 upload
- 基于 device/channel 的 artifact 解析
- episode boundary target sync
- incident 落盘

核心实现：

- [src/lerobot/edge/runtime.py](./src/lerobot/edge/runtime.py)
- [src/lerobot/edge/runner.py](./src/lerobot/edge/runner.py)
- [src/lerobot/edge/deployment_loop.py](./src/lerobot/edge/deployment_loop.py)
- [src/lerobot/edge/model_manager.py](./src/lerobot/edge/model_manager.py)

### Control Plane Report 入口

生成 rollout 快照：

```bash
python -m lerobot.scripts.control_plane_rollout_report --help
```

这个入口会读取：

- registry 状态
- edge incident 文件
- edge model `state.json`

并输出一份汇总 report，包含：

- channel 当前 target artifact
- 已分配设备
- active / pending / previous-active artifact 状态
- incident 与 rollback 统计

核心实现：

- [src/lerobot/control_plane/registry.py](./src/lerobot/control_plane/registry.py)
- [src/lerobot/control_plane/incidents.py](./src/lerobot/control_plane/incidents.py)
- [src/lerobot/control_plane/rollout.py](./src/lerobot/control_plane/rollout.py)

### Unified Cloud Stack 入口

查看统一云端栈参数：

```bash
python -m lerobot.scripts.cloud_stack --help
```

当前支持：

- HTTP ingestion
- HTTP materializer
- 新数据触发 auto release
- `tmux` / `systemd` 常驻部署资产
- `/healthz` 和 `/status` 状态端点

核心实现：

- [src/lerobot/scripts/cloud_stack.py](./src/lerobot/scripts/cloud_stack.py)
- [src/lerobot/scripts/control_plane_auto_release_daemon.py](./src/lerobot/scripts/control_plane_auto_release_daemon.py)
- [src/lerobot/cloud/ingestion.py](./src/lerobot/cloud/ingestion.py)
- [src/lerobot/cloud/materializer.py](./src/lerobot/cloud/materializer.py)

## 测试

当前 edge/control-plane 骨架的测试位于 [tests/edge](./tests/edge)：

- `test_edge_runner.py`
- `test_edge_uploader.py`
- `test_edge_run_local.py`
- `test_release_registry.py`
- `test_model_manager_sync.py`
- `test_deployment_loop.py`
- `test_incident_aggregator.py`
- `test_rollout_status.py`
- `test_control_plane_rollout_report.py`
- `test_http_ingestion.py`
- `test_materializer.py`
- `test_http_materializer.py`
- `test_artifact_builder.py`
- `test_train_and_build.py`
- `test_release_controller.py`
- `test_release_cycle.py`
- `test_auto_release_controller.py`
- `test_cloud_stack.py`

运行当前回归：

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

## 开发工作流

建议的 git 工作流：

1. 保留原仓作为 upstream。
2. 日常开发 push 到自己的 fork。
3. 按主题开分支，而不是在主分支上堆积。
4. 通过按子系统拆分的 PR 并回主仓，而不是一次性超大 diff。

建议的 PR 拆分方式：

- contracts + schemas
- edge runtime + spool + uploader
- registry + deployment loop + rollback
- incidents + rollout reporting
- docs + runbooks

## 实际 roadmap

下一阶段真正高价值的工作，不是继续抽象 edge 内部，而是：

1. 实现真实 ingestion service
2. 将上传 episode materialize 成训练可消费的数据
3. 把 edge 回流数据接进现有训练 pipeline
4. 从 checkpoint 构建真实 deployment artifact
5. 将当前本地文件式 release loop 升级为服务化 controller

## 计划文档

实施计划、阶段状态和剩余缺口见：

- [DRAFT.md](./DRAFT.md)
