# Day19：DPO 训练配置与启动（执行记录）

> 交付物：DPO 训练配置（`configs/exp/*.yaml`）、《DPO训练配置说明.md》、训练启动日志。

## 今天做了什么

1. 写 `template_dpo_qwen.yaml`（以 Week3 SFT 模板为骨架，改 `stage: dpo`），用
   `gen_dpo_configs.py` 渲染出 3 组对比配置 + `experiments.json`。
2. 写 `run_dpo.py`（复用 Week3 `run_experiments.py` 的 VramMonitor/断点续跑/tee，新增 EtaMonitor
   与时间闸），先冒烟（50 样本）验证管线，再全量训练。
3. 写 `collect_dpo_results.py` 汇总 rewards 趋势。

## 关键设计决策

1. **ref_model 隐式等价**：LoRA 下未指定 ref_model 时，LLaMA-Factory 以"禁用 adapter 的同一 policy"
   作参考，而 policy 正是上周最优 SFT 模型，满足任务书要求且省约 6GB 显存。
2. **cutoff_len 从 1024 降到 768**：冒烟实测 cutoff=1024 时峰值显存 23988MiB（逼近 24GB 上限）、
   单步 ~17s；降到 768 后峰值约 13.7GB、单步 ~2.8s。这是本周对"训练时间不能太长"约束的核心优化。
3. **packing 必须关**：SFT 模板里 packing=true，DPO 会破坏成对结构，务必删掉。
4. **时间闸**：`--max-minutes 75` 单组超时终止、`--budget-min 210` 总预算跳过剩余组，配合 EtaMonitor
   实时打印剩余时间。

## 执行方式

```powershell
.\.venv\Scripts\python.exe Week4/code/gen_dpo_configs.py
.\.venv\Scripts\python.exe Week4/code/run_dpo.py --smoke --max-minutes 15
.\.venv\Scripts\python.exe Week4/code/run_dpo.py --max-minutes 75 --budget-min 210
.\.venv\Scripts\python.exe Week4/code/collect_dpo_results.py --copy-logs
```

## 插曲

- 冒烟阶段单步 ~17s 一度让人担心 3 组要跑近 4 小时；实际上那是小数据集（12 步）被固定开销放大的
  假象。全量训练在 cutoff=768 下稳定在 ~2.8s/步，单组约 16 分钟，3 组约 50 分钟。
- 冒烟即验证了 rewards 趋势正确（accuracies 从随机水平上升、chosen>rejected），管线无误后才投全量。

## 明日衔接

Day20 合并最优组、红线安全测试、对齐效果对比。
