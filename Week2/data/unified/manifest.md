# Day6 原始数据归档清单

> 数据源：HuggingFace（经 `官方` 镜像）

| 数据集 | HF 仓库 | 请求 | 下载 | 有效 | 许可 | 说明 |
|---|---|---|---|---|---|---|
| alpaca_gpt4_zh | `llamafactory/alpaca_gpt4_zh` | 2000 | 2000 | 2000 | Apache-2.0 (data: GPT-4 generated) | GPT-4 生成的中文指令-回答对，LLaMA-Factory 官方镜像 |
| coig_pc | `BAAI/COIG-PC-Lite` | 2000 | 2000 | 1976 | 见 BAAI/COIG-PC 数据卡（各子任务不同） | BAAI 中文开放指令通用语料（Lite 精选子集） |
| sharegpt_zh | `shareAI/ShareGPT-Chinese-English-90k` | 1000 | 1000 | 999 | Apache-2.0 | 多轮中文对话（ShareGPT 90k 的中文子集） |

**合计有效样本：4975 条**（已统一为 Alpaca 与 ShareGPT 两种格式）

## 产出文件
- `Week2/data/raw/*.jsonl` —— 各数据集原始子集（未加工，gitignored）
- `Week2/data/unified/alpaca_all.json` —— 统一 Alpaca 格式 `{instruction,input,output}`
- `Week2/data/unified/sharegpt_all.json` —— 统一 ShareGPT 格式 `{conversations:[{from,value}]}`

> 下一步（Day7）：`clean_pipeline.py` 对 `alpaca_all.json` 去噪、去重、截断，产出 ≥1500 条清洗集。