# Week4 Day20 交付：最优 DPO 模型归档说明

> 本文档说明本周最终交付的 DPO 模型是谁、为什么是它、放在哪里。

## 最优配置

| 项 | 值 |
|---|---|
| policy 起点 | `models/Qwen2.5-3B-week3-best-merged`（Week3 最优 SFT） |
| 最优 DPO 组 | **`qwen_dpo_beta0.5_lr5e-6`**（β=0.5, lr=5e-6） |
| 判定依据 | ①Rewards 趋势正确（accuracies 0.974）②安全拒答率 **100%**（唯一过 90% 硬指标的 DPO 组）③对齐主观质量 4.52 不低于 SFT-only 4.50 |

> **为什么不是任务书指定的 β=0.1？** β=0.1 基线组红线拒答率仅 85%、未过硬指标（约束太松，
> 有用性偏好压过了安全性，详见《安全测试记录表.md》与周报 §五）。控制变量实验中的 β=0.5 组
> 拒答率 100%，成为实际交付模型。

## 归档路径

| 产物 | 路径 | 说明 |
|---|---|---|
| DPO LoRA adapter（原始产物） | `saves/week4/qwen/qwen_dpo_beta0.5_lr5e-6/` | 含训练日志，gitignored |
| **合并后完整模型（最终交付）** | `models/Qwen2.5-3B-week4-dpo-merged/` | safetensors 分片 ≤5GB，已生成 |
| 合并配置 | `Week4/configs/merge_best_dpo.yaml` | adapter 路径已指向 β=0.5 组 |

合并命令（Windows 必须用 python -m，见 Week2 Day10 FAQ）：

```powershell
.\.venv\Scripts\python.exe -m llamafactory.cli export Week4/configs/merge_best_dpo.yaml
```

## 验证记录

- Rewards 趋势（β=0.5 组）：`chosen` -0.008→+1.007、`rejected` -0.020→-0.972、`margins` 末 1.979、
  `accuracies` 末 0.974，趋势全部正确。
- 安全拒答率：最优 DPO（β=0.5）在 10 题红线测试上 **100%**（≥90% ✅），且高于 DPO 前 SFT 基线 95%。
- 过度拒绝检查：业务 Prompt biz-05（合法安全科普"网络攻击手段与防范"）正常完整作答，未过度拒绝 ✅。
- 训练成本：14m46s、峰值显存 13.6GB。

## 下周衔接

- 最终 DPO 模型 = `models/Qwen2.5-3B-week4-dpo-merged`，是 SFT→DPO 两阶段对齐后的产物。
- 可作为后续量化部署（AWQ/GPTQ，参考 week1 `export_awq.py`）或进一步评测的输入。
- 若追求更强偏好，可扩充自建安全对、尝试其他偏好损失（如 SimPO/IPO，LLaMA-Factory 已支持
  `pref_loss` 切换）或引入在线采样的偏好数据。
