# 云端训练 + 本地推理执行计划

## 1. 目标

基于当前 LeRobot / HIL-SERL 风格代码，将系统落成三层：

- 云端：训练、评测、模型打包、模型发布
- 本地边缘机：传感器采集、预处理、策略推理、安全约束、机器人执行
- 控制面：数据上传、模型注册、灰度发布、回滚、观测

目标不是先做“最强在线 RL 系统”，而是先做一个：

- 断网不影响机器人闭环
- 模型可版本化发布和回滚
- 数据可稳定回流到训练端
- 能从离线训练平滑演进到近在线闭环

## 2. 非目标

当前阶段不做：

- 实时云端推理
- learner 直接推最新参数到生产机器人
- 多机房 / 多地域容灾
- 复杂权限系统
- 为旧格式保留兼容层

## 3. 关键设计决策

### 3.1 闭环必须本地闭合

实时控制链路固定为：

`camera/state -> local env processor -> local policy runtime -> local action processor -> robot`

该链路不依赖网络，不依赖云端 learner 在线。

### 3.2 训练链路和发布链路分离

- 训练链路产出 checkpoint
- 发布链路产出 deployment artifact
- 只有通过评测和兼容性检查的 artifact 才能进入边缘机

### 3.3 边缘机只拉取“完整推理包”

推理包不是裸权重，必须包含：

- policy weights
- policy config
- env processor config
- action processor config
- dataset stats / normalization stats
- observation schema version
- action schema version
- expected camera keys
- robot compatibility
- runtime ABI version
- training lineage
- eval summary
- artifact digest

### 3.4 发布按 channel 进行，不按单一状态名进行

边缘机不直接“拉 production/latest”，而是绑定 channel：

- `dev`
- `staging`
- `prod`

单台机器人或单类任务可被分配到不同 channel。灰度和回滚都通过 channel 绑定完成。

## 4. 系统边界

### 4.1 本地边缘机职责

- 机器人驱动和相机驱动
- teleop 输入
- observation 组装和预处理
- policy forward
- safety、IK、action clamp、intervention
- episode recorder
- 本地 durable spool
- uploader
- model manager
- watchdog

### 4.2 云端职责

- data ingestion
- dataset materialization
- replay buffer / learner
- reward relabel
- offline eval / sim eval
- artifact build
- registry
- release controller
- metrics and incident sink

### 4.3 控制面职责

- 设备注册
- channel 分配
- artifact promotion
- rollout / rollback
- 兼容性 gate
- 审计和观测

## 5. Runtime Contract

这是实现前必须冻结的 contract。没有这层，后续模块会反复返工。

### 5.1 控制频率

第一版先固定：

- robot control tick: `20 Hz`
- target inference budget: `<= 25 ms`
- hard inference timeout: `40 ms`
- action queue low watermark: `2`
- stale action threshold: `150 ms`

如果目标机器人后续要求更高频率，再重新评估，不先抽象成可配置大系统。

### 5.2 降级策略

本地 watchdog 触发以下动作：

- 单次 forward 超时：丢弃当前 chunk，进入 hold-position
- 连续 3 次 forward 超时：切 safe-stop，等待 teleop 接管
- camera timeout：切 safe-stop
- action queue 空且无法补充：切 hold-position
- model runtime crash：回滚到上一个 stable artifact；若失败则 safe-stop

### 5.3 模型切换时机

只允许在以下时机切换模型：

- episode boundary
- idle state
- 手工 maintenance mode

禁止在 action chunk 执行中途切换。

## 6. 数据契约

### 6.1 Episode 主键

每个 episode 必须生成稳定主键：

- `episode_id`: UUIDv7
- `robot_id`
- `policy_artifact_id`
- `processor_bundle_id`
- `task_id`
- `started_at`

### 6.2 Step 主键

每条 transition 必须包含：

- `episode_id`
- `step_idx`
- `obs_ts`
- `action_ts`
- `teleop_override`
- `reward_raw`
- `reward_relabel_version`
- `done_local`
- `done_train`

说明：

- `done_local` 是执行态事件，用于本地 runtime
- `done_train` 是训练标签，可由云端 relabel 重写
- `reward_raw` 永远保留
- `reward_relabel` 作为新列附加，不覆盖原始数据

### 6.3 上传幂等

上传协议必须支持：

- chunk sequence 编号
- 断点续传
- chunk 去重
- episode commit

只有收到 `episode commit success`，本地 spool 才允许删除对应数据。

## 7. Artifact Contract

### 7.1 Artifact Manifest

每个可部署 artifact 必须带 `manifest.json`，最少包含：

```json
{
  "artifact_id": "policy-prod-00017",
  "base_checkpoint": "ckpt-004231",
  "runtime_abi_version": "v1",
  "observation_schema_version": "v1",
  "action_schema_version": "v1",
  "env_processor_digest": "sha256:...",
  "action_processor_digest": "sha256:...",
  "stats_digest": "sha256:...",
  "policy_digest": "sha256:...",
  "compatible_robot_types": ["koch", "so100"],
  "compatible_camera_layouts": ["front_single_rgb"],
  "eval_summary": {
    "offline_score": 0.0,
    "sim_success_rate": 0.0
  },
  "created_at": "2026-03-27T00:00:00Z"
}
```

### 7.2 兼容性 Gate

边缘机下载 artifact 后，本地必须检查：

- robot type match
- camera layout match
- schema version match
- runtime ABI match
- required processor steps present
- stats present
- digest 校验通过

任一失败直接拒绝激活。

## 8. 发布契约

### 8.1 Artifact 状态

- `draft`
- `candidate`
- `approved`
- `deprecated`

### 8.2 Channel 映射

channel 维护当前目标 artifact：

- `dev -> artifact_x`
- `staging -> artifact_y`
- `prod -> artifact_z`

边缘机只关心：

- 我属于哪个 channel
- 这个 channel 当前目标 artifact 是什么

### 8.3 回滚规则

回滚不是“重发旧状态”，而是：

- 直接把 channel 指回上一个 stable artifact
- 边缘机在下一个安全切换点回切

## 9. 实施阶段

### 9.0 当前进展快照

截至当前代码实现，已经落地的入口和模块包括：

- edge 执行入口：`src/lerobot/scripts/edge_run_local.py`
- control plane 汇总入口：`src/lerobot/scripts/control_plane_rollout_report.py`
- edge 核心模块：`runtime.py`、`runner.py`、`watchdog.py`、`deployment_loop.py`
- 数据回流模块：`episode.py`、`recorder.py`、`spool.py`、`uploader.py`
- 发布与观测模块：`model_manager.py`、`registry.py`、`incidents.py`、`rollout.py`

当前测试覆盖：

- `tests/edge/test_edge_runner.py`
- `tests/edge/test_edge_uploader.py`
- `tests/edge/test_edge_run_local.py`
- `tests/edge/test_release_registry.py`
- `tests/edge/test_model_manager_sync.py`
- `tests/edge/test_deployment_loop.py`
- `tests/edge/test_incident_aggregator.py`
- `tests/edge/test_rollout_status.py`
- `tests/edge/test_control_plane_rollout_report.py`

当前状态判断：

- Phase 0：已完成
- Phase 1：已完成最小稳定版本
- Phase 2：已完成本地闭环版本，云端服务仍未做
- Phase 3：未开始
- Phase 4：已完成最小控制面闭环
- Phase 5：未开始

## Phase 0：冻结契约

目标：把实现前必须一致的接口定死。

当前状态：已完成

交付物：

- runtime contract
- episode schema
- artifact manifest schema
- release channel schema
- 本地目录结构约定

任务：

1. 定义本地 `observation dict` 和 `action dict` 的 canonical schema。
2. 定义 episode/transition 存储格式。
3. 定义 artifact manifest 和 digest 规则。
4. 定义 channel、device registration、artifact assignment 模型。
5. 明确 watchdog 触发条件和降级动作。

验收：

- 文档评审通过
- 不存在 `done/reward` 语义冲突
- 不存在 `staging`/`production` 发布语义冲突

## Phase 1：本地单机稳定推理

目标：机器人在断网情况下可以稳定运行已发布模型。

当前状态：已完成最小版本

交付物：

- local policy runtime
- action queue
- local model manager
- watchdog
- episode recorder

任务：

1. 复用现有 processor pipeline，拆出本地最小 env processor 和 action processor。
2. 基于现有 async inference 思路实现本机 policy runtime。
3. 加入 action chunk queue 和 stale action 检测。
4. 实现 episode recorder，记录 artifact/version/robot config。
5. 实现 watchdog 和 safe-stop。

验收：

- 断网时可持续跑完整 episode
- 模型崩溃时能回退或 safe-stop
- 模型切换仅发生在 episode boundary

当前已实现：

- `EdgeRuntimeContract`、`EdgeWatchdogPolicy`
- `LocalPolicyRuntime` 与 `StaticActionRuntime`
- `EdgeRobotRunner`、`EdgeWatchdog`
- `EdgeDeploymentLoop`
- `edge_run_local.py` dry-run 与多 episode boundary 切换

## Phase 2：数据回流闭环

目标：本地数据稳定上传，云端能 materialize 成训练输入。

当前状态：已完成本地闭环版本

交付物：

- local durable spool
- uploader
- ingestion service
- dataset materializer

任务：

1. 设计 spool 目录和 WAL 规则。
2. 实现 episode chunk upload 和 commit 协议。
3. 实现云端去重和幂等写入。
4. 实现 materializer，将 episode 转成训练数据集。
5. 保留 raw reward 和 relabel reward 双轨字段。

验收：

- 断网恢复后不会重复污染训练集
- 视频和结构化 transition 不错位
- materialized dataset 可直接被现有训练脚本消费

当前已实现：

- `EdgeEpisodeRecord` / `EdgeEpisodeStepRecord`
- `EdgeEpisodeSpool`
- `EdgeEpisodeRecorder`
- `EdgeEpisodeUploader`
- `FilesystemEpisodeIngestionStore`

当前未实现：

- 真实云端 ingestion service
- dataset materializer
- 与现有训练脚本的真实数据集接通

## Phase 3：云端训练与评测

目标：训练端能稳定消费回流数据并产出候选模型。

当前状态：未开始

交付物：

- learner deployment
- offline eval
- sim eval
- artifact builder

任务：

1. 复用现有 learner / buffer / reward model 组件。
2. 接入 demonstrations + online episodes 混合训练。
3. 产出 checkpoint 后自动触发 offline eval。
4. 通过评测的 checkpoint 打包为 candidate artifact。
5. 保存 artifact lineage、评测指标和依赖 digest。

验收：

- 训练任务可重放
- candidate artifact 可追溯到训练数据和代码版本
- 未通过评测的模型不会进入发布通道

## Phase 4：控制面与灰度发布

目标：支持 staging 验证、单机 canary 和 prod 回滚。

当前状态：已完成最小控制面闭环

交付物：

- registry
- release controller
- device/channel mapping
- rollback flow

任务：

1. 实现 artifact registry 和 channel target 表。
2. 实现边缘机定时轮询 channel target。
3. 实现 approve / promote / rollback 操作。
4. 增加设备健康状态和当前运行 artifact 上报。
5. 增加发布审计日志。

验收：

- 单台测试机可自动跟随 `staging`
- `prod` 可在一次操作内回滚到上一版本
- 线上设备运行版本可查询

当前已实现：

- `ReleaseRegistry`
- `EdgeModelManager.sync_to_registry_target`
- `EdgeDeploymentLoop` 的 episode boundary sync
- `model_crash -> rollback_or_safe_stop`
- `FilesystemIncidentSink`
- `IncidentAggregator`
- `RolloutStatusBuilder` / `RolloutStatusStore`
- `control_plane_rollout_report.py`

当前未实现：

- 真正的 release controller 服务
- 设备定时轮询守护进程
- 审计日志服务化
- 线上健康状态远程上报

## Phase 5：近在线闭环

目标：让回流数据自动进入周期训练和受控发布。

当前状态：未开始

交付物：

- scheduled retraining
- auto candidate generation
- gated promotion to staging

任务：

1. 定义训练触发器：按天或按 episode 数。
2. 加入评测阈值和 blocker。
3. 支持自动升 `staging`，人工升 `prod`。
4. 为测试机器人启用近在线更新。

验收：

- 新数据可在固定周期进入训练
- `staging` 自动更新但 `prod` 仍受控
- 线上异常时能快速停止自动推广

## 10. 工作流拆分

### Workstream A：Edge Runtime

范围：

- local env processor
- local action processor
- local runtime
- watchdog
- model manager

优先级：最高

### Workstream B：Data Plane

范围：

- recorder
- spool
- uploader
- ingestion
- materializer

优先级：最高

### Workstream C：Training Plane

范围：

- learner
- reward relabel
- offline eval
- sim eval
- artifact build

优先级：中

### Workstream D：Control Plane

范围：

- registry
- channel mapping
- promote / rollback
- audit / observability

优先级：中

## 11. 建议目录规划

当前实际目录：

```text
src/lerobot/edge/
  __init__.py
  contracts.py
  deployment_loop.py
  episode.py
  incidents.py
  mock_runtime.py
  model_manager.py
  processing.py
  recorder.py
  runner.py
  runtime.py
  spool.py
  uploader.py
  watchdog.py

src/lerobot/control_plane/
  __init__.py
  artifact.py
  incidents.py
  registry.py
  release.py
  rollout.py

src/lerobot/cloud/
  ingestion.py

src/lerobot/scripts/
  edge_run_local.py
  control_plane_rollout_report.py
```

后续保留扩展位：

```text
src/lerobot/cloud/
  materializer.py
  artifact_builder.py
  trainer_bridge.py

src/lerobot/control_plane/
  controller.py
  audit.py
  health.py
```

原则：

- 不把“研究态 learner 逻辑”直接塞进 edge runtime
- 不把“生产态 release 逻辑”混进现有训练脚本

## 12. 风险与阻塞项

### 高风险

1. processor 配置与 runtime 实现漂移
2. reward/done 语义不统一
3. 本地时延超预算导致 action queue 抖动
4. 上传重试导致重复 episode
5. artifact 回滚时 runtime ABI 不兼容

### 对应措施

1. 先冻结 schema 和 manifest，再开始写代码。
2. 保留 raw 与 relabel 双轨字段。
3. 先固定单机器人目标频率，不做过度抽象。
4. 上传协议必须以 episode commit 为删除条件。
5. ABI version 进入 artifact compatibility gate。

## 13. 本周可开工任务

如果现在就开始实现，按顺序做：

1. 写 `artifact manifest schema` 文档和样例。
2. 写 `episode schema` 文档和样例。
3. 从现有 processor 中裁出 edge 最小 runtime pipeline。
4. 搭一个单机 local policy runtime，先只支持 eager。
5. 做 watchdog + safe-stop。
6. 做本地 recorder + spool。
7. 设计 uploader 的 chunk + commit 协议。

以上 1-7 已完成最小版本。当前更合理的下一步是：

1. 实现真实 ingestion service 和 dataset materializer。
2. 把回流 episode 接到现有训练脚本输入。
3. 实现 artifact builder，把 checkpoint 打包成标准 deployment artifact。
4. 将 release controller 从本地文件流升级成服务化状态机。

## 14. 完成定义

这项工作的第一阶段完成，不以“代码很多”为准，而以以下结果为准：

- 机器人可在断网状态下稳定执行已发布模型
- 每个 episode 可稳定落盘并回传
- 云端能产出带 manifest 的 candidate artifact
- staging 和 prod 的发布语义清晰且可回滚
- reward / done / schema / ABI 没有歧义

按当前代码状态，这个“第一阶段完成定义”已经基本满足，但仍缺两项真正产品化能力：

- 云端 materializer / trainer / artifact builder 的接通
- 服务化的 release controller 与远程健康上报

## 15. 结论

当前最合理的落地路径不是直接追求在线 RL 全自动闭环，而是：

1. 先把 edge runtime 做稳
2. 再把数据回流做对
3. 再把训练产物变成标准 artifact
4. 最后用 channel 化发布把 staging / prod 管起来

这个顺序和当前代码库的现状一致，也最符合真实机器人系统的风险控制方式。
