# 第 4 周：DPO 偏好对齐

> 环境：Windows 11 + RTX 4090 (24GB)，训练环境 `.venv/`（Python 3.12）。
> 承接第三周的最优 SFT 模型 `models/Qwen2.5-3B-week3-best-merged`（DPO 的 policy 起点），
> 本周做**偏好数据构造 → DPO 训练（控制变量对比）→ 红线安全测试 → 对齐效果对比 → 归档**。
> 核心目标：让模型学会"分辨好坏"，把**安全拒答率打到 ≥90%**（本周唯一硬性指标）。

## 目录结构

```
Week4/
├── code/
│   ├── make_self_built_pairs.py   Day18 手写 221 条自建偏好对（内容即代码，可复现）
│   ├── build_preference_data.py   Day18 开源对(UF+DPO-Zh) + 自建对 → 合并/校验/注册/统计图
│   ├── gen_dpo_configs.py         Day19 从模板生成 3 个 DPO 配置 + experiments.json
│   ├── run_dpo.py                 Day19 批量训练（ETA 监控 + 单组/总预算时间闸）
│   ├── collect_dpo_results.py     Day19-20 汇总 rewards/loss/耗时 → 对比表 + 日志归档
│   ├── safety_eval.py             Day20 10 高危 Prompt 逐模型作答 → 判定表；--tally 算拒答率
│   ├── compare_alignment.py       Day20 5 业务 Prompt：SFT-only vs SFT+DPO 并排 + 空白评分表
│   └── plot_rewards.py            Day21 rewards 曲线图（chosen↑/rejected↓/margins/accuracies）
├── configs/
│   ├── template_dpo_qwen.yaml     DPO 模板（含 ${...} 占位符，不可直接训练）
│   ├── exp/                       生成的 3 个可运行配置 + experiments.json
│   └── merge_best_dpo.yaml        Day20 最优 DPO adapter 合并归档
├── data/
│   ├── self_built_pairs.json      自建 221 条（安全80/事实40/完整36/有用35/格式30，进 git）
│   ├── safety_prompts.json        10 条红线 Prompt（10 类风险）
│   ├── business_prompts.json      5 条业务 Prompt（对应 5 种偏好类型）
│   └── dpo/                       生成物：dpo_pairs.json(1221) + pairs_meta.json + dataset_info.json
├── docs/                          Day17~21 每日工作说明
└── deliverables/
    ├── 偏好数据构造指南.md          Day17：DPO 原理 + 5 类偏好 + 安全红线 + 反过度拒绝
    ├── 偏好数据集统计.md + pref_*.png   Day18：来源/类型/长度统计
    ├── DPO训练配置说明.md + logs/       Day19：配置要点 + rewards 趋势判定规则
    ├── DPO实验结果汇总.md               Day20：3 组 rewards 对比（脚本自动生成）
    ├── 安全测试记录表.md + 安全判定记录.csv   Day20：红线测试（硬指标）
    ├── 对齐效果对比表.md + 对齐主观评分.csv   Day20：SFT vs DPO
    ├── rewards_*.png                    Day21：rewards 曲线
    ├── 最优DPO模型说明.md               Day20：归档路径 + 下周衔接
    ├── FAQ.md                          本周踩坑记录
    └── 第4周_DPO偏好对齐报告.md          Day21：周报
```

## 运行顺序（在 SenceTime_Week1/ 根目录）

```powershell
# ① Day18 构建偏好数据集（CPU，约 5–10 分钟；开源对首跑需下载，之后走本地缓存）
.\.venv\Scripts\python.exe Week4/code/make_self_built_pairs.py
.\.venv\Scripts\python.exe Week4/code/build_preference_data.py

# ② Day19 生成配置 → 冒烟 → 全量（带时间闸；训练前关闭 Chrome/微信/Steam 见下）
.\.venv\Scripts\python.exe Week4/code/gen_dpo_configs.py
.\.venv\Scripts\python.exe Week4/code/run_dpo.py --smoke                       # 50 样本冒烟
.\.venv\Scripts\python.exe Week4/code/run_dpo.py --max-minutes 75 --budget-min 210
.\.venv\Scripts\python.exe Week4/code/collect_dpo_results.py --copy-logs

# ③ Day20 合并最优组（Windows 必须用 python -m）+ 安全与对齐测试
.\.venv\Scripts\python.exe -m llamafactory.cli export Week4/configs/merge_best_dpo.yaml
.\.venv\Scripts\python.exe Week4/code/safety_eval.py            # 生成答卷 + 空白判定 CSV
#   ↑ 人工填写 Week4/deliverables/安全判定记录.csv 的「判定」列后：
.\.venv\Scripts\python.exe Week4/code/safety_eval.py --tally    # 回收 → 拒答率
.\.venv\Scripts\python.exe Week4/code/compare_alignment.py

# ④ Day21 rewards 曲线图
.\.venv\Scripts\python.exe Week4/code/plot_rewards.py
```

## 实验设计一图流

- policy 起点 = Week3 最优 SFT 合并模型；ref_model 隐式等价（LoRA 下禁用 adapter 的同一模型）。
- 基线 = **β=0.1 / lr=5e-6**（任务书指定组），被 A/B 两组共享；最终交付因安全指标改用 β=0.5 组。
- 组A（β）：0.1 vs 0.5；组B（lr）：5e-6 vs 1e-5。共 **3 次训练**。
- 固定：seed 42、`week4_dpo_pairs`(1221 条)、r=32/α=64、等效 batch 8、ep 2、cutoff 768、bf16。

## 关键工程要点（详见 FAQ.md）

- **packing 必须关**：序列打包是 SFT 专用，会破坏 chosen/rejected 成对结构。
- **cutoff_len 768**：DPO 单步跑 4 遍前向（policy+ref × chosen+rejected），实测 1024 时峰值显存
  逼近 24GB；降到 768 后峰值约 13.7GB、单步 ~2.8s，3 组约 50 分钟。
- **ranking: true**：数据集注册表缺此字段会被当普通 SFT 读，DPO 失效。
- **训练前关闭桌面应用**（Chrome/微信/Steam/VS Code）：Week3 实测 WDDM 时间片争抢可拖慢 65%。
- **导出用 `python -m llamafactory.cli`**：`.exe` 在含撇号路径下段错误（Week2 Day10 FAQ）。

## 验收标准对照

| # | 验收 | 状态 |
|---|---|---|
| ❶ | 偏好数据 ≥300 条且质量合格 | ✅ 1221 条（开源 1000 + 自建 221），5 类全覆盖，校验通过 |
| ❷ | DPO Rewards 指标趋势正确 | ✅ 三组 chosen↑/rejected↓/margins↑/accuracies→0.92-0.97 |
| ❸ | 高危 Prompt 安全拒答率 ≥90%（硬指标） | ✅ 最优 β=0.5 组 **100%**（β=0.1 基线仅 85% 未达标，见下） |
| ❹ | 周报提交 | ✅《第4周_DPO偏好对齐报告.md》 |

> ★ 本周核心发现：**任务书指定的 β=0.1 反而把安全拒答率从 SFT 的 95% 拉低到 85%**——偏好数据
> 以"有用性"为主导时，β 太小会让 policy 偏离 SFT 过远、有用性压过安全性。控制变量里的 β=0.5
> 组（约束更紧）守住了拒答能力，拒答率 100%，成为最终交付模型。详见周报 §五。
