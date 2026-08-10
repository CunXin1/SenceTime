# 第 2 周：数据清洗 & 首次 SFT 训练

> 环境：Windows 11 + RTX 4090 (24GB)，Python 3.12 venv（`.venv/`）。
> 承接第一周的 LLaMA-Factory 微调管线，本周做**数据获取 → 清洗 → 正式 SFT → 合并测试**。
> 训练两个 3B 基座做跨架构对比：`Qwen2.5-3B-Instruct` 与 `Llama-3.2-3B-Instruct`（均 LoRA）。

## 目录结构

```
Week2/
├── code/
│   ├── download_data.py     Day6 下载3数据集 + 统一 Alpaca/ShareGPT 格式 + 归档清单
│   ├── download_models.py   下载 Qwen2.5-3B / Llama-3.2-3B 基座（HF优先，回退ModelScope）
│   └── clean_pipeline.py     Day7 清洗：去HTML/控制字符/空值/超长截断 + SimHash去重 + 统计图
├── configs/
│   ├── qwen3b_lora_sft.yaml  Day8 Qwen 逐行注释配置
│   └── llama3b_lora_sft.yaml Day8 Llama 逐行注释配置（差异：template=llama3）
├── data/
│   ├── raw/                  各数据集原始子集（gitignored）
│   ├── unified/             统一格式 alpaca_all/sharegpt_all + manifest 归档清单
│   └── clean/               清洗后 alpaca_clean/sharegpt_clean + dataset_info.json
└── deliverables/            清洗统计图(clean_funnel/length_dist) + 报告
```

## 运行顺序（在 SenceTime_Week1/ 根目录）

```bash
# 环境已在 .venv，命令用 .venv/Scripts/python.exe 直接调用（无需 activate）

# ① 下模型（HF 优先，Llama gated 自动回退 ModelScope）
.venv/Scripts/python.exe Week2/code/download_models.py

# ② Day6 下数据 + 统一格式
.venv/Scripts/python.exe Week2/code/download_data.py

# ③ Day7 清洗（用真实 Qwen 分词器计 token 更准）
.venv/Scripts/python.exe Week2/code/clean_pipeline.py --tokenizer models/Qwen2.5-3B-Instruct

# ④ Day8 训练（★ 训练前会与你确认；两个基座各跑一次）
.venv/Scripts/llamafactory-cli.exe train Week2/configs/qwen3b_lora_sft.yaml
.venv/Scripts/llamafactory-cli.exe train Week2/configs/llama3b_lora_sft.yaml

# ⑤ Day9 监控 + 合并 + 测试
tensorboard --logdir saves/
.venv/Scripts/llamafactory-cli.exe export Week2/configs/<merge>.yaml
```

## 验收标准对照

| # | 验收 | 状态 |
|---|---|---|
| ❶ | 清洗脚本可独立运行 | ✅ clean_pipeline.py 三级分词回退，纯脚本可跑 |
| ❷ | 训练集 ≥1500 条且格式正确 | ✅ 4684 条，Alpaca+ShareGPT 双格式 |
| ❸ | SFT 训练 loss 稳定下降 | ✅ Qwen 1.58→1.22 / Llama 1.58→1.38 |
| ❹ | 合并后模型对话质量优于基座 | ✅ 更简洁完整、贴合训练风格 |
| ❺ | 周报提交 | ✅ 第2周_数据与SFT入门报告.md + FAQ.md |
```
