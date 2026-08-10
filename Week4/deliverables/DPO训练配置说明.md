# Week4 Day19 交付：DPO 训练配置说明

> 说明本周 3 组 DPO 对比实验的配置要点、控制变量设计，以及**训练前预先约定**（pre-register）
> 的 Rewards 趋势判定规则——这是验收 ❷ 的依据。

## 一、policy 与 ref_model

| 项 | 值 | 说明 |
|---|---|---|
| policy 起点 | `models/Qwen2.5-3B-week3-best-merged` | 第 3 周最优 SFT 合并模型 |
| ref_model | 隐式 = 禁用 adapter 的同一 policy | 见下 |

任务书要求"ref_model 指向上周最优 SFT"。在 `finetuning_type: lora` 下，LLaMA-Factory 的
`create_ref_model()`（`LLaMA-Factory/src/llamafactory/train/trainer_utils.py`）在**未显式指定
ref_model 时，以"禁用 LoRA 旁路的同一个 policy"作参考**。而这里的 policy 正是上周最优 SFT 模型，
因此语义上已满足要求，同时省下约 6GB 显存（不必再加载第二份完整模型）。配置里保留了一行注释掉的
`# ref_model:` 并写明这层等价关系。

## 二、控制变量实验矩阵（3 组）

沿用 Week3 的控制变量法：基线只训一次、被两组共享，每组只改一个变量。

| run_id | 组 | pref_beta | learning_rate | 说明 |
|---|---|---|---|---|
| `qwen_dpo_beta0.1_lr5e-6` | A+B（基线） | 0.1 | 5.0e-6 | 任务书指定组 |
| `qwen_dpo_beta0.5_lr5e-6` | A | 0.5 | 5.0e-6 | 更强 KL 约束 |
| `qwen_dpo_beta0.1_lr1e-5` | B | 0.1 | 1.0e-5 | 更大学习率 |

> 事后结果（Day20）：β=0.1 基线组安全拒答率仅 85%、未过 90% 硬指标，**最终交付改用 β=0.5 组
> （拒答率 100%）**。详见《安全测试记录表.md》与周报 §五。这印证了做 β 对照组的必要性。

**固定项**：policy、`lora_rank=32`/`lora_alpha=64`（沿用 Week3 最优秩）、`pref_loss=sigmoid`、
`pref_ftx=0`、`num_train_epochs=2`、`gradient_accumulation_steps=8`（等效 batch 8）、
`cutoff_len=768`、`packing=false`、`seed=42`、`bf16=true`、`dataloader_num_workers=0`。

## 三、关键参数为什么这么设

1. **`pref_beta`（β）**：控制 policy 偏离 ref 的允许程度（KL 约束强度）。β 小=约束松、学得快、
   偏好强但易遗忘通用能力；β 大=约束紧、更稳但偏好信号弱。任务书基线取 0.1，额外跑 0.5 做对照。
2. **`learning_rate`**：DPO 比 SFT 低一个量级（5e-6 起）。policy 已是对齐良好的 SFT 模型，
   大 lr 会破坏已学能力。额外跑 1e-5 看是否加速收敛。
3. **`cutoff_len=768`**：DPO 单步要跑 policy+ref × chosen+rejected 共 4 遍前向，序列越长显存和
   耗时翻倍增长。实测 1024 时峰值显存逼近 24GB 上限；768 兼顾覆盖率、显存与速度。
4. **`packing=false`**：序列打包是 SFT 专用优化，会把多条样本拼成一条，破坏 chosen/rejected 的
   成对结构——DPO 必须关闭。
5. **`num_train_epochs=2`**：偏好数据 1221 条，2 轮足以让 rewards 曲线收敛，更多轮易过拟合。

## 四、★ Rewards 趋势判定规则（训练前预先约定，验收 ❷ 依据）

DPO 训练过程主要看四条 rewards 曲线（键名见
`LLaMA-Factory/src/llamafactory/train/dpo/trainer.py`）。**趋势正确才算训练有效**：

| 指标 | 含义 | 期望趋势 |
|---|---|---|
| `rewards/chosen` | chosen 的隐式奖励 β·log(π/π_ref) | **上升** |
| `rewards/rejected` | rejected 的隐式奖励 | **下降** |
| `rewards/margins` | chosen − rejected | **单调扩大** |
| `rewards/accuracies` | chosen 奖励 > rejected 的比例 | **上升 → 0.9+** |

**额外健康度约定**：`rewards/chosen` 不应大幅、持续转负。轻微转负常见（policy 在 chosen 上略低于
ref 但仍显著高于 rejected 即可）；若大幅转负说明 β 太小、policy 跑离 ref 太远，需调大 β。

自动判定由 `collect_dpo_results.py` 完成：对每条指标取 first/last 比较，符合期望标 ✅、相反标 ⚠️，
四项全 ✅ 记该组"趋势正确"。

## 五、时间预算与监控（回应"训练时间不能太长"）

- `run_dpo.py` 内置 `EtaMonitor`：每 30s 读 `trainer_log.jsonl` 末行打印步数进度与预计剩余时间，
  并把第 ~10% 步的剩余时间快照记入 `run_meta.json` 的 `eta_at_10pct`（"跑之前就知道要跑多久"）。
- 时间闸：`--max-minutes 75`（单组超时终止）、`--budget-min 210`（累计超预算跳过剩余组）。
- 实测（cutoff 768，RTX 4090）：约 290 步/组、~2.8s/步、单组约 16 分钟，3 组约 50 分钟。

## 六、启动命令

```powershell
.\.venv\Scripts\python.exe Week4/code/gen_dpo_configs.py
.\.venv\Scripts\python.exe Week4/code/run_dpo.py --smoke                          # 冒烟
.\.venv\Scripts\python.exe Week4/code/run_dpo.py --max-minutes 75 --budget-min 210
.\.venv\Scripts\python.exe Week4/code/collect_dpo_results.py --copy-logs
```
