# 第 3 周：SFT 超参优化与模型评估

> 环境：Windows 11 + RTX 4090 (24GB)，训练环境 `.venv/`（Python 3.12），
> OpenCompass 独立环境 `.venv-oc/`。
> 承接第二周的 SFT 管线，本周做**控制变量对比实验 → 自动化评估 → 盲测打分 →
> OpenCompass 客观评测 → 最优模型归档**（为第四周 DPO 做输入）。
> 实验对象：`Qwen2.5-3B-Instruct`，共 4 组对比实验
>（扩展矩阵可用 `gen_configs.py --full --models qwen,llama` 生成）。

## 目录结构

```
Week3/
├── code/
│   ├── gen_configs.py       Day11 从模板生成 14 个实验 yaml + experiments.json 清单
│   ├── run_experiments.py   Day12-13 批量顺序训练（耗时/峰值显存记录、断点续跑）
│   ├── collect_results.py   Day12-13 汇总 loss/耗时/显存 → 实验结果汇总.md + 日志归档
│   ├── eval_harness.py      Day14 基座+PEFT 热挂 adapter，20 固定题批量作答
│   ├── make_scorecard.py    Day14 生成盲测材料（匿名答卷/空白打分CSV/保密映射）
│   ├── make_radar.py        Day14 打分回收 → 5 维雷达图 + 加权总分 + 最优标记
│   └── run_opencompass.md   Day15 OpenCompass 操作手册（Windows 排障 + LF eval 兜底）
├── configs/
│   ├── template_qwen.yaml / template_llama.yaml   实验模板（含 ${...} 占位符）
│   ├── exp/                 生成的 14 个可运行配置 + experiments.json
│   └── merge_best_*.yaml    Day16 最优模型合并归档配置
├── data/
│   └── eval_questions.json  20 道固定高难度题（数学7/推理7/代码6，含参考要点）
├── docs/
│   ├── Day11~16 每日工作说明（设计/执行记录/操作指南）
│   └── WSL2迁移评估.md      Week4 前的基础设施决策材料
└── deliverables/
    ├── 实验计划表.md          Day11：实验矩阵 + 假设 + 判定规则
    ├── 实验结果汇总.md        Day12-13：14 组对照表（脚本自动生成）
    ├── logs/                 Day12-13：各组原始训练日志归档
    ├── 人工评分卡.md          Day14：5 维评分标准（30/25/20/15/10）
    ├── 盲测答卷.md / 盲测打分记录.csv / 盲测映射表.md   Day14 盲测材料
    ├── radar_*.png / 盲测得分汇总.md                   Day14 雷达图与得分
    ├── OpenCompass评测分数表.md                        Day15
    ├── 最优模型说明.md                                 Day16：归档路径 + DPO 衔接
    ├── FAQ.md                                          本周踩坑记录（8 问）
    └── 第3周_SFT优化与评估报告.md                       周报
```

## 运行顺序（在 SenceTime_Week1/ 根目录）

```powershell
# ① Day11 生成实验配置（默认 qwen 精简 5 组；--full 为 7 组）
.\.venv\Scripts\python.exe Week3/code/gen_configs.py

# ② Day12-13 批量实验（约 2h；先冒烟再全量；支持断点续跑）
.\.venv\Scripts\python.exe Week3/code/run_experiments.py --smoke
.\.venv\Scripts\python.exe Week3/code/run_experiments.py
.\.venv\Scripts\python.exe Week3/code/collect_results.py --copy-logs

# ③ Day14 自动化评估 + 盲测
.\.venv\Scripts\python.exe Week3/code/eval_harness.py
.\.venv\Scripts\python.exe Week3/code/make_scorecard.py --models <入围模型逗号分隔>
#   ↑ 把 盲测答卷.md + 盲测打分记录.csv 发给打分人，映射表先保密
.\.venv\Scripts\python.exe Week3/code/make_radar.py

# ④ Day15 OpenCompass（详见 Week3/code/run_opencompass.md，含兜底方案）

# ⑤ Day16 合并归档最优模型（Windows 必须用 python -m）
.\.venv\Scripts\python.exe -m llamafactory.cli export Week3/configs/merge_best_qwen.yaml
.\.venv\Scripts\python.exe -m llamafactory.cli export Week3/configs/merge_best_llama.yaml
```

## 实验设计一图流

- 基线 = Week2 配置 + 固定种子：`r8 / lr1e-4 / ep3`，被 A/B/C 三组共享。
- 组A 秩：8 vs 32（α≡2r）；组B 学习率：1e-4 vs 2e-4；组C 轮数：3 vs 5。
- 共 **4 次训练**（1 个共享基线 + 每组 1 个变体）。
- 固定：seed 42、`week2_sft_alpaca`(4684 条)、等效 batch 16、cosine+warmup10%、bf16、packing。

## 验收标准对照

| # | 验收 | 状态 |
|---|---|---|
| ❶ | 完成至少 3 组有效对比实验 | A（秩）/ B（学习率）/ C（轮数）× 双模型 |
| ❷ | 有明确数据支撑最优选择 | 实验结果汇总表 + 盲测雷达图 + OpenCompass 三重证据 |
| ❸ | OpenCompass 跑通且出分 | 见 OpenCompass评测分数表.md |
| ❹ | 周报提交 | 第3周_SFT优化与评估报告.md |
