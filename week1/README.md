# 第 1 周:环境搭建 & Qwen2.5 大模型导论

> 实习第 1 周项目。Mac(M4 Pro / MPS)上完成环境、推理、架构分析、分词实验、LoRA 微调;
> 7B QLoRA + AWQ 留待 4090。

## 目录结构

```
Week1/
├── code/                    # 可运行脚本(统一从 Week1/ 根目录运行)
│   ├── download_models.py       下载模型到 ./models/
│   ├── inference/
│   │   ├── inference.py         通用推理(Qwen/Llama),含 4 步管线
│   │   ├── inference_gemma4.py  Gemma-4 专用推理(多模态,AutoProcessor)
│   │   └── compare_inference.py 4 模型横评
│   ├── analysis/
│   │   └── count_params.py      参数量分布统计
│   └── finetune/
│       ├── finetune.py         LoRA/QLoRA 训练(改 MODEL 切 7B)
│       ├── lf_lora.yaml        训练基础配置
│       ├── compare_finetune.py 微调前后对比
│       └── export_awq.py       合并 LoRA + AWQ 压缩(AWQ 需 4090)
├── docs/                    # 说明 / 讲解 / 笔记
│   ├── TODO.md                  执行清单
│   ├── 任务与约束.md            任务总览 + Mac/4090 取舍
│   ├── Python文件说明.md        脚本说明
│   ├── 推理管线详解.md          推理 4 步详解
│   ├── 现代LLM设计_中英.md      RoPE/GQA/SwiGLU 等
│   └── 训练语料方案.md          语料格式与来源
├── deliverables/            # 交付物(按主题)
│   ├── 环境/                   conda_list + 环境验证
│   ├── 推理输出/               4模型推理输出(按问题分组)
│   ├── 架构分析/               设计哲学 + config逐字段分析 + 架构报告 + 4模型对比 + 参数统计
│   ├── 分词实验/               tokenizer_experiments.ipynb + BPE_vs_SentencePiece.md
│   ├── 微调/                   LoRA微调报告 + 前后对比 + loss曲线
│   ├── 第1周完整交付.md         汇总所有交付的单一总文档
│   └── 周报/                   第1周总结报告
├── models/                  # 4 个模型权重(已 gitignore,勿删)
└── saves/                   # LoRA adapter 输出(35MB 各)
```

## 运行约定(重要)

脚本内用 `./models/...` 相对路径,所以有**两条铁律**:

1. **必须从 `Week1/` 根目录运行**(不是从脚本所在目录)
2. **必须先激活对应 conda 环境**:推理/分析用 `llm_exp`,微调用 `llama_factory`

> ⚠️ **不要在 VSCode 直接点"运行 ▶"** —— 它默认在文件所在目录跑、且可能用错解释器,会报"找不到 models / 模块导入失败"。请用下面的终端方式。

```bash
# ① 推理 / 分析(llm_exp 环境)
conda activate llm_exp
cd ~/SenceTime/Week1                                   # 关键:回到根目录
python code/download_models.py                        # 下载模型
python code/inference/inference.py                    # 默认 Qwen2.5(加 --template 看 ChatML)
python code/inference/inference_gemma4.py             # Gemma-4
python code/inference/compare_inference.py            # 4 模型横评
python code/analysis/count_params.py ./models/Qwen3-4B

# ② 微调(llama_factory 环境)
conda activate llama_factory
cd ~/SenceTime/Week1
python code/finetune/finetune.py                      # LoRA 训练(alpaca_zh_demo)
python code/finetune/finetune.py identity_penguin     # 指定语料
python code/finetune/compare_finetune.py              # 微调前后对比
python code/finetune/export_awq.py                    # 合并 LoRA(AWQ 量化需 4090)
```

> 若一定要用 VSCode:右下角 "Select Interpreter" 选对环境,并确保工作区打开的是 `Week1/` 根目录。

## 进度对照(验收标准)

| # | 验收 | 状态 |
|---|---|---|
| ❶ | CUDA 可用 | 🟡 待 4090(Mac 用 MPS) |
| ❷ | 基座模型能对话 | ✅ 完成 |
| ❸ | 口头解释 RoPE / GQA | ✅ 完成 |
| ❹ | LLaMA-Factory demo 无报错 | ✅ **Mac MPS 上 LoRA 训练成功** |
| ❺ | 周报提交 | ✅ 完成 |

## 环境

- `llm_exp`(py3.10):推理/分析,MPS 版 PyTorch + transformers + LangChain
- `llama_factory`(py3.11):LLaMA-Factory 训练 + TensorBoard
- `opencompass`(py3.10):OpenCompass 评测框架
- 详见 `docs/任务与约束.md`
