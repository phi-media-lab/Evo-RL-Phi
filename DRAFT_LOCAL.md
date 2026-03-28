# 本地端 OpenPI Backend 集成完整设计

## 1. 目标

将 `/Users/fbsh/ane-openpi/openpi` 提供的本地推理能力，以**新 runtime backend** 的形式接入当前 Evo-RL 本地端推理栈。

目标不是替换当前本地端系统，而是：

- 复用 Evo-RL 已有的 edge runtime shell
- 让 OpenPI / ANE / CoreML / MLX hybrid 成为可选推理 backend
- 保持现有 spool / upload / rollout / rollback / incident / report 体系不变

## 2. 总体结论

最优集成边界是：

- `Evo-RL` 负责 edge runtime shell
- `ane-openpi` 负责特定策略家族与特定硬件推理 backend

这意味着：

- 不重写本地执行闭环
- 不另造第二套发布和数据回流体系
- 不把 `ane-openpi` 当成完整 edge 系统

而是把它接成：

- `openpi_pytorch`
- `openpi_coreml`
- `openpi_hybrid_bridge_v1`

这类 runtime backend。

## 3. 现状

### 3.1 Evo-RL 当前已经具备什么

当前仓库已经有完整本地端外壳：

- runtime contract
- local runner
- watchdog
- recorder / spool / uploader
- model manager
- registry target sync
- boundary-based activation
- rollback
- incident sink

关键模块：

- [src/lerobot/edge/runtime.py](./src/lerobot/edge/runtime.py)
- [src/lerobot/edge/runner.py](./src/lerobot/edge/runner.py)
- [src/lerobot/edge/deployment_loop.py](./src/lerobot/edge/deployment_loop.py)
- [src/lerobot/edge/model_manager.py](./src/lerobot/edge/model_manager.py)
- [src/lerobot/scripts/edge_run_local.py](./src/lerobot/scripts/edge_run_local.py)

### 3.2 当前仓库并不是完全不懂 OpenPI

当前仓库已经内置 OpenPI 家族的 LeRobot direct port：

- `PI0Policy`
- `PI05Policy`
- `PI0FastPolicy`

并且已有原始 OpenPI 与 LeRobot 端口的对齐测试：

- [tests/policies/pi0_pi05/test_pi0_original_vs_lerobot.py](./tests/policies/pi0_pi05/test_pi0_original_vs_lerobot.py)
- [tests/policies/pi0_pi05/test_pi05_original_vs_lerobot.py](./tests/policies/pi0_pi05/test_pi05_original_vs_lerobot.py)

因此，接入 `ane-openpi` 的主要价值不是“获得 OpenPI 模型语义”，而是：

- 获得 `ane-openpi` 中更激进的本地推理 backend
- 获得 ANE / CoreML / MLX hybrid 路径

### 3.3 当前 `ane-openpi` 提供什么

`/Users/fbsh/ane-openpi/openpi` 当前能提供三类能力：

1. policy / checkpoint 加载能力
2. websocket policy server
3. CoreML / MLX / hybrid bridge 的实验性推理路径

其中最有价值的是：

- PyTorch OpenPI policy
- rollout CoreML artifact
- Bridge V1 / MLX prefix feed 原型

但这些能力目前更偏：

- benchmark / export / probe / experiment

还不是稳定 edge runtime API。

### 3.4 当前已落地进展

到当前这个阶段，这份设计前半段已经基本落成代码，而不是只停留在 draft：

- `EdgePolicyRuntime` 协议已经落地
- `runner` / `deployment_loop` / `model_manager` 已改为依赖 runtime protocol / factory
- `OpenPIWebsocketRuntime` 已实现并通过 `mock robot -> runner -> spool` 回归
- `OpenPIInProcessRuntime` 已实现，并支持 `policy_ref -> ane-openpi policy` 正式加载
- `OpenPIObservationAdapter` 已支持：
  - `flat_state`
  - `aloha_raw`
  - `openpi_raw`
- `EdgeModelManager` 已能识别：
  - `policy_config.type = "openpi"`
  - `metadata.runtime.backend = "websocket" | "in_process" | "coreml" | "hybrid_bridge_v1"`
- `FilesystemOpenPIArtifactBuilder` 已实现，`openpi` artifact 已可 build / load / activate
- `openpi` artifact 已能进入现有：
  - `EdgeRobotRunner`
  - `EdgeDeploymentLoop`
  - `EdgeModelManager.activate_artifact()`

另外，真实 smoke 已经完成了三层验证：

- `resolve_openpi_policy_ref(...)` 成功加载真实 `pi0_aloha_sim`
- `policy.infer(...)` 成功返回真实 action chunk
- `openpi artifact -> model_manager.activate_artifact() -> runner -> spool` 已真实跑通

并且，`hybrid_bridge_v1` 的真实 feed 路径也已经完成了最小验证：

- `pi0_aloha_sim` 已成功转换为 PyTorch checkpoint
- 真实 `aloha_sim_row000000` observation 已成功构造出 Bridge V1 feed
- 当前一次真实结果表明：
  - `resolve_feed_builder_s` 约为数百秒
  - `build_feed_s` 约为数百秒
- 首轮最重成本是 `load_pytorch(...)`
- 为了避免每次 feed 构造都重复首轮加载，当前 loader 已补：
  - `feed builder cache`
  - `model context cache`
- `mlx_full_prefix_hostbridge` 已在当前 Evo-RL edge shell 中真实跑通到：
  - artifact build
  - `EdgeModelManager.activate_artifact()`
  - `EdgeRobotRunner`
  - spool
- 当前主要问题已经从“能不能集成”切到“首轮延迟是否可接受”

### 3.5 当前可迁移运行入口

为了把当前两套代码库迁移到更高算力的 Mac，本仓库已经补了可直接复用的本地运行入口：

- 环境初始化：
  - [scripts/setup_openpi_edge_env.sh](./scripts/setup_openpi_edge_env.sh)
- in-process smoke：
  - [scripts/run_openpi_inprocess_smoke.sh](./scripts/run_openpi_inprocess_smoke.sh)
- hybrid smoke：
  - [scripts/run_openpi_hybrid_smoke.sh](./scripts/run_openpi_hybrid_smoke.sh)
- 迁移/运行说明：
  - [OPENPI_LOCAL_RUNTIME.md](./OPENPI_LOCAL_RUNTIME.md)

这组入口的目的，是把：

- `Evo-RL`
- `ane-openpi`
- converted checkpoint
- rollout `mlpackage`
- sample observation

收成一套可迁移、可复现的本地推理发行方式，而不是继续依赖当前机器的临时路径。

## 4. 当前设计问题

现在最大的阻碍不是 observation adapter，而是 **runtime 层仍然硬编码为单实现**。

当前几个核心耦合点：

- [src/lerobot/scripts/edge_run_local.py](./src/lerobot/scripts/edge_run_local.py)
  `runtime` 直接是 `LocalPolicyRuntimeConfig`
- [src/lerobot/edge/runner.py](./src/lerobot/edge/runner.py)
  `runtime` 显式类型是 `LocalPolicyRuntime`
- [src/lerobot/edge/model_manager.py](./src/lerobot/edge/model_manager.py)
  `activate_artifact()` / `build_runtime_config()` 直接实例化 `LocalPolicyRuntime`

所以如果不先解耦 runtime，后面无论接 `ane-openpi` 还是别的 backend，都会继续复制 `LocalPolicyRuntime` 分支逻辑。

## 5. 设计原则

### 5.1 统一外壳，不统一内部推理实现

统一的东西保留在 Evo-RL：

- deployment loop
- watchdog
- spool / upload
- artifact registry / rollout / rollback
- incident / report

可变的东西下沉到 runtime backend：

- observation preprocessing
- prompt / token 处理
- actual policy inference
- prefix cache / rollout kernel / CoreML bridge

### 5.2 继续使用统一 artifact contract

不引入第二套 artifact 系统。

统一使用现有：

- [src/lerobot/control_plane/artifact.py](./src/lerobot/control_plane/artifact.py)

但要扩展 backend-aware metadata。

### 5.3 先做 runtime 解耦，再做 OpenPI 集成

正确顺序是：

1. 先把 edge runtime 从单实现抽成协议
2. 再加 OpenPI backend
3. 再做 artifact / model manager 扩展

### 5.4 websocket 只用于 smoke

`ane-openpi` 的 websocket server 可以用于：

- integration smoke
- backend 插拔性验证

但不作为最终正式运行形态。

## 6. 目标架构

目标架构如下：

```text
edge_run_local
  -> EdgeDeploymentLoop
    -> EdgeRobotRunner
      -> EdgePolicyRuntime (protocol)
        -> LocalPolicyRuntime
        -> StaticActionRuntime
        -> OpenPIWebsocketRuntime
        -> OpenPIPolicyRuntime
```

模型激活路径：

```text
ReleaseRegistry
  -> EdgeModelManager
    -> RuntimeFactory
      -> backend-specific runtime config
      -> backend-specific runtime instance
```

artifact 仍统一：

```text
ArtifactManifest
  + policy_config.type
  + metadata.runtime.backend
  + metadata.runtime_assets
  + metadata.observation_contract
```

## 7. Runtime 抽象设计

### 7.1 新协议

新增：

- `EdgePolicyRuntime`
- `EdgePolicyRuntimeConfig`

建议位置：

- `src/lerobot/edge/runtime_protocol.py`

最小协议：

```python
class EdgePolicyRuntime(Protocol):
    @property
    def queue_size(self) -> int: ...
    def reset(self) -> None: ...
    def warmup(self, observation: dict[str, Any]) -> torch.Tensor: ...
    def maybe_refill_action_queue(self, observation: dict[str, Any]) -> torch.Tensor | None: ...
    def pop_next_action(self, observation: dict[str, Any]) -> torch.Tensor: ...
```

### 7.2 现有实现迁移

现有：

- `LocalPolicyRuntime`
- `StaticActionRuntime`

都改成实现这个协议。

### 7.3 Runtime factory

新增：

- `RuntimeFactory`
- `make_runtime_from_artifact(...)`

建议位置：

- `src/lerobot/edge/runtime_factory.py`

factory 根据 artifact manifest 里的 `policy_config.type` 和 `metadata.runtime.backend` 选择 runtime 实现。

## 8. Artifact 设计扩展

### 8.1 保持现有字段

以下字段继续保留为硬 gate：

- `runtime_abi_version`
- `observation_schema_version`
- `action_schema_version`
- `compatible_robot_types`
- `compatible_camera_layouts`

### 8.2 新增 backend-aware metadata

对 OpenPI backend，建议最少增加：

```json
{
  "policy_config": {
    "type": "openpi"
  },
  "metadata": {
    "runtime": {
      "backend": "openpi_pytorch",
      "runtime_mode": "in_process"
    },
    "observation_contract": {
      "image_keys": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
      "requires_image_mask": true,
      "prompt_mode": "default_prompt",
      "state_dtype": "float32"
    },
    "runtime_assets": {
      "checkpoint_dir": "policy",
      "coreml_model_path": "runtime/model.mlpackage",
      "bridge_feed_contract_path": "runtime/feed_contract.json"
    }
  }
}
```

### 8.3 为什么要加 runtime assets

仅有 `policy_path` 不够，因为 OpenPI backend 可能依赖：

- checkpoint
- CoreML package
- prompt / tokenizer 资产
- prefix cache contract
- export metadata

这些都需要作为激活前的显式依赖，而不是隐含在脚本逻辑里。

## 9. Observation Contract 设计

### 9.1 不从零设计 adapter

优先抽取现有仓库里已经存在的 OpenPI-compatible preprocessing 逻辑，而不是新写一套从头适配器。

来源包括：

- 当前仓库里的 `PI0Policy` / `PI05Policy` preprocessing 路径
- 原始 OpenPI vs LeRobot 对齐测试

### 9.2 Edge-safe observation adapter

新增：

- `OpenPIObservationAdapter`

建议位置：

- `src/lerobot/edge/openpi_processing.py`

职责：

- 将 edge observation 转成 OpenPI `Observation`
- 对齐 image keys
- 填充 image masks
- 注入 prompt / token fields
- 处理 state dtype / shape

### 9.3 adapter 输出 contract

输出至少保证：

- `image`
- `image_mask`
- `state`
- 可选 `tokenized_prompt`
- 可选 `tokenized_prompt_mask`
- 可选 `token_ar_mask`
- 可选 `token_loss_mask`

并形成明确 schema 文档。

## 10. 集成分阶段

### Phase 0：runtime 解耦

目标：

- 去掉 `LocalPolicyRuntime` 的硬绑定

交付物：

- `EdgePolicyRuntime` 协议
- `RuntimeFactory`
- runner / deployment loop / model manager 改为依赖协议

验收：

- 现有 `LocalPolicyRuntime` 路径不回归
- `StaticActionRuntime` 仍通过当前所有 edge 测试

当前状态：

- 已完成
- `runtime_protocol.py` 已引入
- `runner.py` / `deployment_loop.py` / `model_manager.py` 已切到 runtime protocol / runtime factory

### Phase 1：OpenPI websocket smoke

目标：

- 验证 OpenPI backend 可以接进 edge shell

交付物：

- `OpenPIObservationAdapter`
- `OpenPIWebsocketRuntime`

说明：

- 仅用于 integration smoke
- 不进入最终正式路径

验收：

- `mock robot -> OpenPI websocket runtime -> spool`
- runner / recorder / deployment loop 正常

当前状态：

- 已完成
- `OpenPIWebsocketRuntime` 已实现
- 已有回归覆盖 `mock robot -> websocket runtime -> runner -> spool`

### Phase 2：OpenPI in-process backend

目标：

- 提供正式 backend

交付物：

- `OpenPIPolicyRuntime`
- backend 选择：`openpi_pytorch`

说明：

- 直接 import `openpi`
- 不走 websocket

验收：

- 本地 in-process 推理可替换 `LocalPolicyRuntime`
- action chunk 能被现有 queue 机制消费

当前状态：

- 已完成最小版
- `policy_ref -> resolve_openpi_policy_ref(...)` 已接入真实 `ane-openpi`
- 真实 `pi0_aloha_sim` checkpoint load smoke 已通过
- 真实 `policy.infer(...)` smoke 已通过
- `aloha_raw` observation contract 已进入正式代码路径

### Phase 3：OpenPI artifact builder

目标：

- 让 OpenPI runtime 能通过统一 artifact 激活

交付物：

- `OpenPIArtifactBuilder`
- model manager 按 `policy_config.type == "openpi"` 分流

建议位置：

- `src/lerobot/cloud/openpi_artifact_builder.py`

验收：

- `artifact manifest -> model manager -> runtime factory -> OpenPI runtime`

当前状态：

- 已完成最小版
- `FilesystemOpenPIArtifactBuilder` 已实现
- `cloud_build_openpi_artifact.py` 已实现
- `openpi artifact -> model manager.build_runtime_config() -> activate_artifact()` 已通过回归
- `openpi artifact -> runner -> spool` 真实 smoke 已通过

### Phase 4：CoreML / hybrid backend

目标：

- 将 `ane-openpi` 的 CoreML / hybrid 路径收敛成正式 backend

交付物：

- `OpenPICoreMLRuntime`
- `OpenPIHybridBridgeRuntime`

说明：

- 不直接复用 benchmark script 作为 runtime
- 必须先抽成稳定模块 API

当前状态：

- 已完成最小版 runtime contract 与 runtime shell
- `OpenPICoreMLRuntime` / `OpenPIHybridBridgeRuntime` 已进入正式代码路径
- `EdgeModelManager.build_runtime_config()` 已能正式识别：
  - `backend = "coreml"`
  - `backend = "hybrid_bridge_v1"`
- runtime assets 缺失现在会被 `validate_artifact()` 显式拦住
- `hybrid_bridge_v1` 已不再只是匿名 metadata，而是正式 runtime config / factory 路径
- `ane-openpi` 的 Bridge V1 feed 构造已接入当前仓库 loader：
  - 真实 `pi0_aloha_sim_pytorch` checkpoint 已成功生成
  - 真实 sample observation 已成功构造出 Bridge V1 feed
- 当前已确认真实 Hybrid feed 的首轮成本主要在：
  - `load_pytorch(...)`
  - 而不是 observation adapter 或 feed packing
- 为此已经落地两层缓存：
  - `feed builder cache`
  - `model context cache`
- 当前仍未完成的部分是：
  - 真实 `model.mlpackage` predictor 接入
  - CoreML / hybrid rollout 的端到端执行 smoke

验收：

- 能在本地端边界切换
- 能纳入 watchdog / rollback / incident

### Phase 5：失败注入与回滚验证

目标：

- 确认新 backend 在 release loop 中可被安全托管

重点验证：

- warmup 失败
- activate 失败
- runtime crash
- observation contract mismatch
- runtime assets 缺失

验收：

- `activate -> run -> crash -> rollback -> incident`
- 与现有 `model_crash -> rollback_or_safe_stop` 语义一致

当前状态：

- 未完成
- rollback 主链仍在，但还没有专门为 `openpi` backend 做故障注入验证

## 11. 代码改造点

### 11.1 `edge_run_local.py`

当前：

- 直接使用 `LocalPolicyRuntimeConfig`

改造后：

- 接受 `EdgeRuntimeConfig`
- 或 `runtime_backend` + backend-specific config

### 11.2 `runner.py`

当前：

- `runtime: LocalPolicyRuntime`

改造后：

- `runtime: EdgePolicyRuntime`

### 11.3 `deployment_loop.py`

当前：

- 通过 `isinstance(self.runtime, LocalPolicyRuntime)` 判断是否 activate / rollback

改造后：

- 不再依赖具体类
- 使用 runtime capability 或 backend type

### 11.4 `model_manager.py`

当前：

- `build_runtime_config()` 只返回 `LocalPolicyRuntimeConfig`
- `activate_artifact()` 直接 new `LocalPolicyRuntime`

改造后：

- `build_runtime_spec()`
- `activate_artifact()` 通过 runtime factory 创建实例

### 11.5 `artifact_builder.py`

当前：

- 默认假设 `policy/config.json` + LeRobot `from_pretrained`

改造后：

- 保持现有 builder
- 新增 `OpenPIArtifactBuilder`

## 12. 风险

### 12.1 误以为当前 OpenPI direct port 足以替代 `ane-openpi`

风险：

- 混淆“OpenPI 模型语义”与“ANE / CoreML backend 能力”

应对：

- 明确集成 `ane-openpi` 的目标是 backend，不是模型语义本身

### 12.2 将 benchmark script 直接当 runtime

风险：

- runtime 语义不稳定
- CLI / 文件输出逻辑污染 edge runtime

应对：

- 先抽库接口，再接 runtime

### 12.3 artifact contract 不完整

风险：

- artifact 可以被加载，但不能安全 warmup / activate

应对：

- 将 backend assets / observation contract 写入 manifest

### 12.4 rollback 语义被破坏

风险：

- 新 backend crash 后无法回退

应对：

- 单独做失败注入验证阶段

## 13. 推荐实施顺序

按优先级：

1. runtime 解耦
2. preprocessing / observation contract 抽取
3. websocket smoke
4. in-process OpenPI PyTorch backend
5. OpenPI artifact builder
6. CoreML / hybrid backend
7. failure injection / rollback validation

按当前实际进度看：

1. runtime 解耦：完成
2. preprocessing / observation contract 抽取：完成最小版
3. websocket smoke：完成
4. in-process OpenPI PyTorch backend：完成最小版
5. OpenPI artifact builder：完成最小版
6. CoreML / hybrid backend：完成最小版，真实 MLX hybrid smoke 已打通
7. failure injection / rollback validation：下一步

## 14. 当前阶段的明确决策

本阶段先不做：

- 直接让 websocket 进入正式部署
- 直接把 benchmark script 接进 edge loop
- 重写第二套 artifact 系统
- 跳过 runtime 解耦直接硬塞 OpenPI backend

本阶段必须先做：

- `EdgePolicyRuntime` 协议
- `RuntimeFactory`
- backend-aware artifact contract
- edge-specific observation contract

这些项现在已经完成，因此当前阶段的明确决策应切到：

- 不再继续扩展 websocket 路径
- `OpenPICoreMLRuntime` / `OpenPIHybridBridgeRuntime` 的 runtime contract、class、factory 已完成
- 真实 Bridge V1 feed 已经打通到 PyTorch checkpoint + sample observation
- CoreML / hybrid 当前最需要推进的已不再是 schema，而是：
  - 首轮延迟阶段 profiling
  - persistent runtime / preload
  - 真实 `model.mlpackage` predictor 接入
  - failure injection / rollback validation

## 15. 结论

这项集成的正确方向不是：

- “把 `ane-openpi` 整体塞进当前本地端”

而是：

- “把 Evo-RL 现有本地端作为统一运行外壳”
- “把 `ane-openpi` 收敛成一个或多个正式 runtime backend”

只有这样，才能同时保留当前仓库已经完成的：

- rollout / rollback
- recorder / spool / upload
- incident / report
- channel / registry / deployment loop

并把 OpenPI / ANE / CoreML / MLX 的本地推理能力安全纳入现有 edge runtime 体系。
