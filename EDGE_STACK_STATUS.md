# 本地端推理栈状态清单

这份文档专门描述 Evo-RL 当前本地端推理栈的实现状态，用于回答两个问题：

- 本地端已经做到了什么程度
- 本地端还缺哪些能力才算产品化

这里的“本地端”主要指：

- `src/lerobot/edge/`
- `src/lerobot/scripts/edge_run_local.py`

不包含云端训练、artifact build、control-plane orchestration 本身。

## 1. 当前结论

当前本地端推理栈已经达到：

- 最小可用工程骨架已完成
- 关键执行闭环已完成
- 已具备和云端 / 控制面联通的能力

但还没有达到：

- 真实生产环境下的长期稳定部署水平
- 多设备运行的完整运维和审计水平
- 实机充分验证后的产品化完成状态

一句话判断：

本地端已经完成“最小 edge inference stack”，但还没有完成“实机产品化 edge runtime”。

## 2. 已完成能力

### 2.1 Runtime Contract 已冻结

当前已经有稳定的执行约束，见 [src/lerobot/edge/contracts.py](./src/lerobot/edge/contracts.py)：

- 控制频率
- inference budget
- hard timeout
- stale action threshold
- watchdog policy
- runtime ABI / observation schema / action schema version

这意味着本地端不再依赖隐式约定，而是有明确 runtime contract。

### 2.2 本地 Runtime 已具备最小执行能力

核心实现见：

- [src/lerobot/edge/runtime.py](./src/lerobot/edge/runtime.py)
- [src/lerobot/edge/mock_runtime.py](./src/lerobot/edge/mock_runtime.py)

当前已经具备：

- local policy runtime
- dry-run runtime
- action queue
- action chunk 消费
- 基于本地 processor bundle 的最小推理路径

这意味着本地端已经能独立完成最小推理执行，而不是依赖云端在线推理。

### 2.3 Runner / Watchdog / Deployment Loop 已打通

核心实现见：

- [src/lerobot/edge/runner.py](./src/lerobot/edge/runner.py)
- [src/lerobot/edge/watchdog.py](./src/lerobot/edge/watchdog.py)
- [src/lerobot/edge/deployment_loop.py](./src/lerobot/edge/deployment_loop.py)

当前已经具备：

- `robot -> observation -> runtime -> action -> recorder`
- inference timeout 检查
- stale action 检查
- queue empty 检查
- `model_crash -> rollback_or_safe_stop`
- episode boundary 模型切换

这意味着本地执行不只是“能 forward 一次”，而是已经有完整的最小运行闭环。

### 2.4 数据落盘链路已完成

核心实现见：

- [src/lerobot/edge/episode.py](./src/lerobot/edge/episode.py)
- [src/lerobot/edge/recorder.py](./src/lerobot/edge/recorder.py)
- [src/lerobot/edge/spool.py](./src/lerobot/edge/spool.py)
- [src/lerobot/edge/uploader.py](./src/lerobot/edge/uploader.py)

当前已经具备：

- episode / step schema
- durable spool
- sealed / uploaded 生命周期
- chunk + commit 上传
- filesystem sink
- HTTP 上传路径

这意味着本地端数据已经能稳定形成“可回流的结构化训练输入”。

### 2.5 模型管理和发布切换已完成最小版本

核心实现见：

- [src/lerobot/edge/model_manager.py](./src/lerobot/edge/model_manager.py)
- [src/lerobot/control_plane/registry.py](./src/lerobot/control_plane/registry.py)
- [src/lerobot/control_plane/artifact.py](./src/lerobot/control_plane/artifact.py)

当前已经具备：

- artifact manifest 校验
- compatibility gate
- active / pending / previous_active 状态
- device -> channel -> target artifact 解析
- `sync_to_registry_target()`
- rollback 到 previous active artifact

这意味着本地端已经能消费控制面定义的发布目标，而不是只能手工指定模型目录。

### 2.6 Incident 记录已完成

核心实现见：

- [src/lerobot/edge/incidents.py](./src/lerobot/edge/incidents.py)

当前已经具备：

- watchdog incident 记录
- rollback action 记录
- episode 级 incident 文件输出

这意味着本地端已经有稳定的异常痕迹，后续控制面可以消费这些状态。

### 2.7 CLI 入口已具备最小可用性

核心实现见：

- [src/lerobot/scripts/edge_run_local.py](./src/lerobot/scripts/edge_run_local.py)

当前已经支持：

- dry-run
- mock robot
- 本地 spool
- HTTP ingestion upload
- device/channel 解析
- 多 episode 运行
- episode boundary sync

这意味着本地端不是只有库级能力，而是已经有可直接运行的入口。

## 3. 已验证能力

当前已经通过测试或真实环境 smoke 验证的本地端能力包括：

- runner 执行
- spool / uploader
- edge_run_local dry-run
- registry target resolution
- model manager sync / rollback
- deployment loop
- HTTP upload 路径

相关测试见：

- [tests/edge/test_edge_runner.py](./tests/edge/test_edge_runner.py)
- [tests/edge/test_edge_uploader.py](./tests/edge/test_edge_uploader.py)
- [tests/edge/test_edge_run_local.py](./tests/edge/test_edge_run_local.py)
- [tests/edge/test_release_registry.py](./tests/edge/test_release_registry.py)
- [tests/edge/test_model_manager_sync.py](./tests/edge/test_model_manager_sync.py)
- [tests/edge/test_deployment_loop.py](./tests/edge/test_deployment_loop.py)
- [tests/edge/test_http_ingestion.py](./tests/edge/test_http_ingestion.py)

这说明本地端已经跨过“只有结构设计”的阶段，进入“已有可运行实现”的阶段。

## 4. 未完成能力

下面这些能力仍未完成，因此当前不能把本地端定义为“产品化完成”。

### 4.1 实机长时间稳定性验证不足

当前还缺：

- 多小时或多天持续运行验证
- 高频 episode 切换下的稳定性验证
- 长时间 spool / upload / rollback 混合运行验证

现在更多是结构正确和最小功能正确，而不是长期稳定性已证明。

### 4.2 真实机器人 / 真实相机链路还未充分打磨

当前还缺：

- 更完整的 camera timeout / reconnect 策略
- 真实 action processor / safety processor 的实机调优
- 真实机器人驱动链路的长跑验证

也就是说，当前 runner 闭环已经存在，但离“充分适配真实机器人生产场景”还有距离。

### 4.3 Edge 常驻运行形态还不够正式

当前有脚本入口，但还缺：

- 更正式的 edge daemon
- daemon 生命周期管理
- restart policy
- 本地运行日志标准化

云端这边已经有 `tmux/systemd` 资产，本地端还没有形成同等级部署资产。

### 4.4 健康上报还没有形成正式能力

当前虽然有 incident，但还缺：

- device heartbeat
- current runtime status
- last upload timestamp
- queue depth / camera health / runtime state 的周期上报

这意味着控制面还不能系统性判断“设备是否在线且健康”。

### 4.5 artifact 本地生命周期管理不完整

当前已经有 manifest 校验和 activate / rollback，但还缺：

- 下载缓存策略
- 旧 artifact 清理策略
- warmup 生命周期管理
- 磁盘占用控制

这部分在单机 smoke 中不是问题，但在长期运行里会成为运维问题。

### 4.6 审计与运维细节未完成

当前还缺：

- 本地运行审计
- 更细粒度操作日志
- 远程状态追踪
- 多设备统一排障入口

## 5. 当前阶段判断

如果把本地端推理栈按成熟度分层，可以这样判断：

- contract 和核心抽象：已完成
- 本地执行主路径：已完成最小版本
- 数据落盘与回流接口：已完成最小版本
- 模型切换与回滚：已完成最小版本
- 实机产品化和运维：未完成

因此更准确的说法是：

- 本地端骨架：已完成
- 本地端最小闭环：已完成
- 本地端产品化：未完成

## 6. 对项目整体的意义

当前本地端已经不再是项目的主要短板。

它已经足够支撑：

- 云端 ingestion / materializer 对接
- artifact build / release contract 对接
- control-plane rollout / rollback 对接

所以项目下一阶段的主要重心不应继续放在“补更多本地端基础对象”，而应放在：

- release policy
- approval gate
- training / eval gate
- health visibility
- control-plane 产品化

只有在你准备做真实机器人上线或长跑实机验证时，本地端才会重新回到最高优先级。

## 7. 下一步建议

如果仍然要继续补本地端，建议优先级如下：

1. edge heartbeat / health reporting
2. edge 常驻 daemon 与部署资产
3. 真实机器人 / 相机链路长跑验证
4. artifact cache / cleanup 生命周期管理
5. 更细的本地日志与审计

如果以整体项目优先级看，则更建议先推进：

1. `prod` release policy 和 approval gate
2. training / artifact evaluation gate
3. control-plane health aggregation

## 8. 结论

当前本地端推理栈已经达到“最小可用 edge inference stack”的水平。

它已经足够支撑当前云边闭环继续向前推进，但还不足以宣称“本地端产品化完成”。
