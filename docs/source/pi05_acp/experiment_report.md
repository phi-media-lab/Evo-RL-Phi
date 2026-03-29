# PI05 Evo-RL ACP 实验报告

## 1. 实验目标

本实验的目标是基于前期 AMD ROCm 配置经验，在当前机器上完成以下验证：

1. 配置并验证 ROCm 后端的 LeRobot / Evo-RL 运行环境
2. 跑通 `pi05` 的真实推理
3. 跑通 `pistar06 -> value-infer -> ACP indicator -> pi05 policy train` 的 Evo-RL ACP workflow
4. 将实验从 smoke test 逐步扩大到可持续训练规模，并验证 checkpoint 稳定性

## 2. 实验环境

### 2.1 硬件与基础环境

- GPU: `AMD Instinct MI300X VF`
- ROCm: 可用
- Python: `3.12.3`
- Torch: `2.6.0+rocm7.0.2`
- LeRobot: `0.4.4`

### 2.2 代码库

- 主代码仓: `Evo-RL-Phi`
- OpenPI 源码: `openpi`

### 2.3 使用的 Python 环境

正式实验使用的隔离环境：

```text
/root/phi-media-lab/.venvs/pi05-openpi-ssp
```

该环境中额外完成了：

- `transformers==4.53.2`
- OpenPI `transformers_replace` 补丁覆盖
- editable install 的 `Evo-RL-Phi`

## 3. 关键兼容性问题与解决方案

### 3.1 `pi05` 对 OpenPI patched transformers 的依赖

问题：

- 原环境中的 `transformers 4.57.6` 无法满足 OpenPI 版 `pi05`
- `GemmaRMSNorm.forward(cond=...)` 不存在
- `transformers.models.siglip.check` 兼容性脆弱

解决：

- 单独构建 patched venv
- 安装 `transformers==4.53.2`
- 覆盖 OpenPI 的 `transformers_replace`

结果：

- `pi05` 的 forward 和 `select_action` 在 ROCm 上可运行

### 3.2 gated tokenizer 访问

问题：

- `google/paligemma-3b-pt-224` 初始无权限

解决：

- 更新 Hugging Face 账号权限
- 本地重新 `hf auth login`

结果：

- 真实 tokenizer 加载成功
- `lerobot/pi05_base` 真实推理成功

### 3.3 `torchcodec` 与 ROCm 数据解码不兼容

问题：

- DataLoader 侧视频解码默认走 `torchcodec`
- 当前环境中 CPU worker 路径不可用，导致视频帧解码失败

解决：

- 全部数据相关命令显式指定：

```bash
--dataset.video_backend=pyav
```

结果：

- `LeRobotDataset` 可稳定读取图像帧
- `value-train` / `value-infer` / `policy train` 全部可运行

### 3.4 数据集缺少 quantile stats

问题：

- `maxbeau/XLeRobot` 的 `observation.state` 只有 `min/max/mean/std/count`
- 缺失 `q01/q99`
- 默认 `QUANTILES` 归一化会报错

解决：

- 在 value / policy 训练中统一覆盖：

```bash
--normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
```

结果：

- 成功绕过 quantile 依赖
- 当前所有实验阶段都采用该方案

### 3.5 数据集缺少 `episode_success`

问题：

- `maxbeau/XLeRobot` 的 episode metadata 中不存在 `episode_success`

解决：

- value 相关流程统一使用：

```bash
--dataset.default_success=failure
```

结果：

- 可完成 value target 构造
- 但当前 value supervision 语义仍然较弱，属于工程可跑通而非最优标签质量

## 4. 本地代码修改

本实验为跑通整条链路，对 `Evo-RL-Phi` 做了最小兼容修改。

### 4.1 `pi05` 相关

[`src/lerobot/policies/pi05/configuration_pi05.py`](../../../../src/lerobot/policies/pi05/configuration_pi05.py)

- 新增 `tokenizer_name`

[`src/lerobot/policies/pi05/processor_pi05.py`](../../../../src/lerobot/policies/pi05/processor_pi05.py)

- 从硬编码 tokenizer 切换到 `config.tokenizer_name`

[`src/lerobot/policies/pi05/modeling_pi05.py`](../../../../src/lerobot/policies/pi05/modeling_pi05.py)

- 放宽 `siglip.check` 的依赖
- 增加 tied-weight fallback，兼容 `lerobot/pi05_base` 权重结构
- 为 OpenPI 专用的 `GemmaRMSNorm(cond=...)` 和 `_gated_residual` 增加 fallback 兼容
- 将 `select_action` 的推理路径切换为“无 KV cache 的全前缀重算”，避开 cache-mask 长度错位

### 4.2 value inference 相关

[`src/lerobot/configs/value.py`](../../../../src/lerobot/configs/value.py)

- 为 `ValueInferenceDatasetConfig` 增加 `video_backend`

[`src/lerobot/scripts/lerobot_value_infer.py`](../../../../src/lerobot/scripts/lerobot_value_infer.py)

- 将 `dataset.video_backend` 传给 `LeRobotDataset`

## 5. 数据集与模型

### 5.1 数据集

实验数据集：

```text
maxbeau/XLeRobot
```

本地缓存根路径：

```text
/root/.cache/huggingface/lerobot/maxbeau/XLeRobot
```

已确认特性：

- 总帧数：`1486`
- episode 数：`5`
- task：`pick up the block`
- 包含视频键：
  - `observation.images.front_cam`
  - `observation.images.hand_cam`
- 不包含：
  - `episode_success`
  - state quantile stats

### 5.2 模型

value model:

```text
pistar06
```

policy model:

```text
lerobot/pi05_base
```

## 6. 实验阶段与结果

### 6.1 Smoke 阶段

目标：

- 验证真实 `pi05` 推理
- 验证 1-step value train / value infer / policy train

结果：

- `pi05` 真实推理成功
- `pistar06` 1-step 训练成功
- `value-infer` 成功写回：
  - `complementary_info.value_smoke`
  - `complementary_info.advantage_smoke`
  - `complementary_info.acp_indicator_smoke`
- `pi05` 1-step ACP 训练成功

### 6.2 Pilot 阶段

脚本：

[`scripts/experiments/pi05_acp/run_pi05_acp_pilot.sh`](../../../../scripts/experiments/pi05_acp/run_pi05_acp_pilot.sh)

配置：

- value train: `5` steps
- value infer: 全量
- policy train: `5` steps

结果：

- 全链路成功
- value checkpoint 成功
- policy checkpoint 成功

输出目录：

- `outputs/value_train/pi05_acp_pilot`
- `outputs/train/pi05_acp_policy_pilot`

### 6.3 Stage1

脚本：

[`scripts/experiments/pi05_acp/run_pi05_acp_stage1.sh`](../../../../scripts/experiments/pi05_acp/run_pi05_acp_stage1.sh)

配置：

- value train: `20` steps
- value infer: 全量
- policy train: `20` steps

结果：

- value 训练成功，checkpoint 于 `10/20`
- value infer 成功写回 `*_stage1`
- policy 训练成功，checkpoint 于 `10/20`

输出目录：

- `outputs/value_train/pi05_acp_stage1`
- `outputs/train/pi05_acp_policy_stage1`

### 6.4 Stage2

脚本：

[`scripts/experiments/pi05_acp/run_pi05_acp_stage2.sh`](../../../../scripts/experiments/pi05_acp/run_pi05_acp_stage2.sh)

配置：

- value train: `100` steps
- value infer: 全量
- policy train: `100` steps

结果：

- value 训练成功，checkpoint 于 `50/100`
- value infer 成功写回 `*_stage2`
- policy 训练成功，checkpoint 于 `50/100`

输出目录：

- `outputs/value_train/pi05_acp_stage2`
- `outputs/train/pi05_acp_policy_stage2`

### 6.5 Stage3

脚本：

[`scripts/experiments/pi05_acp/run_pi05_acp_stage3.sh`](../../../../scripts/experiments/pi05_acp/run_pi05_acp_stage3.sh)

配置：

- value train: `500` steps
- value infer: 全量
- policy train: `500` steps

结果：

- value 训练成功，checkpoint 于 `250/500`
- value infer 成功写回 `*_stage3`
- policy 训练成功，checkpoint 于 `250/500`
- policy 训练过程中 loss 呈下降趋势

关键日志：

```text
step:200 ... loss:0.350 ...
step:400 ... loss:0.245 ...
```

输出目录：

- `outputs/value_train/pi05_acp_stage3`
- `outputs/train/pi05_acp_policy_stage3`

### 6.6 全量 Rerun

为避免覆盖历史产物，又增加了一次完整 rerun。

脚本：

[`scripts/experiments/pi05_acp/run_pi05_acp_full_rerun.sh`](../../../../scripts/experiments/pi05_acp/run_pi05_acp_full_rerun.sh)

统一后缀：

```text
rerun_20260329
```

执行范围：

- `smoke`
- `pilot`
- `stage1`
- `stage2`
- `stage3`

结果：

- 五个阶段全部成功完成
- `stage3 value train` 成功保存 `250/500`、`500/500` checkpoint
- `stage3 value infer` 成功写回：
  - `complementary_info.value_stage3_rerun_20260329`
  - `complementary_info.advantage_stage3_rerun_20260329`
  - `complementary_info.acp_indicator_stage3_rerun_20260329`
- `stage3 policy train` 成功保存 `250/500`、`500/500` checkpoint
- `stage3 policy loss` 再次从约 `0.350` 下降到约 `0.245`

输出目录：

- `outputs/value_train/pi05_acp_stage3_rerun_20260329`
- `outputs/value_infer/pi05_acp_stage3_rerun_20260329`
- `outputs/train/pi05_acp_policy_stage3_rerun_20260329`

## 7. 关键观测

### 7.1 真实 `pi05` 推理已通过

已确认以下链路可用：

- gated tokenizer 加载
- `lerobot/pi05_base` 权重加载
- `select_action` 真实推理

说明当前机器上的 `pi05` 不是“只可训练不可推理”的状态。

补充验证：

```bash
cd <REPO_ROOT>
pytest -q tests/training/test_acp_pi05_prompt_pipeline.py tests/policies/pi0_pi05/test_pi05.py
```

结果：

```text
4 passed, 1 warning in 41.51s
```

这说明 `ACP prompt pipeline`、`pi05 forward` 和 `pi05 select_action` 当前都已通过关键测试。

### 7.2 ACP workflow 已完整跑通

已成功完成：

1. `pistar06` 训练
2. `value-infer`
3. 写回 ACP 标签
4. `pi05` 带 ACP 标签训练

说明 `Evo-RL-Phi` 的 `pi05` ACP pipeline 在本机是闭环可执行的。

### 7.3 训练规模已验证到 500 steps

当前已验证的最大训练规模：

- value train: `500` steps
- policy train: `500` steps

且中途 checkpoint 与最终 checkpoint 均正常。

### 7.4 当前实验仍是“工程可跑通”，不等于“最优训练设置”

原因：

- 使用了 `MEAN_STD` 覆盖，而非数据原生 quantiles
- `episode_success` 缺失，只能把默认 success 设为 `failure`
- 数据集只有 `5` 个 episode，规模很小

因此目前更准确的结论是：

- 工程链路稳定
- 训练过程稳定
- 但数据语义质量和训练规模还不适合直接拿来评估最终策略效果

## 8. 生成的脚本与文档

### 8.1 脚本

- `scripts/experiments/pi05_acp/run_pi05_acp_smoke.sh`
- `scripts/experiments/pi05_acp/run_pi05_acp_pilot.sh`
- `scripts/experiments/pi05_acp/run_pi05_acp_stage1.sh`
- `scripts/experiments/pi05_acp/run_pi05_acp_stage2.sh`
- `scripts/experiments/pi05_acp/run_pi05_acp_stage3.sh`
- `scripts/experiments/pi05_acp/run_pi05_acp_full_rerun.sh`

### 8.2 文档

- `docs/source/pi05_acp/execution_plan.md`
- `docs/source/pi05_acp/runbook.md`

## 9. 风险与限制

### 9.1 数据标签质量

`episode_success` 缺失导致 value targets 的语义较弱。当前方式适合工程验证，不适合严肃评估。

### 9.2 数据归一化不是最优配置

当前使用 `MEAN_STD` 是为兼容旧数据。若补齐 quantile stats，理论上更应回归到 `QUANTILES`。

### 9.3 数据规模过小

`5` 个 episode 只能验证训练管线，无法支持可靠的策略泛化结论。

### 9.4 仍需真实 rollout 评估

当前实验主要验证训练与推理链路，没有完成真实机器人闭环 rollout 评估。

### 9.5 `select_action` 当前偏向正确性优先

为修复 patched Gemma 环境中的 cache-mask 长度错位，当前 `pi05` 推理采用“无 KV cache 的全前缀重算”路径。

这意味着：

- 当前 `select_action` 已恢复正确性
- 但单次推理开销会高于理想的缓存推理实现

## 10. 结论

本实验已经完成以下结论性验证：

1. 当前机器上的 `ROCm + MI300X` 可以稳定运行 `Evo-RL-Phi`
2. `pi05` 在 OpenPI patched transformers 环境下可完成真实推理
3. `pistar06 -> ACP label writeback -> pi05 ACP training` 的完整 workflow 已跑通
4. 训练规模已从 smoke test 扩展到 `500-step + checkpoint`，说明该链路具备中等规模训练稳定性

因此，当前系统状态可以定义为：

```text
pi05 的 Evo-RL ACP workflow 已在 ROCm 环境中完成端到端工程验证，并具备继续放大训练规模的条件
```

## 11. 下一步建议

优先顺序建议如下：

1. 给数据集补 `episode_success`
2. 给数据集补 quantile stats
3. 基于更高质量数据启动 `1000+` steps 长训
4. 对 `stage3` 或后续 checkpoint 做真实机器人 rollout 评估
5. 如果需要长期运行，整理一份正式训练脚本并加入日志/监控策略
