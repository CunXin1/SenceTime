"""Qwen LoRA / QLoRA 微调(基于 LLaMA-Factory)。

改 MODEL 切换 1.5B / 7B;改 QLORA 切换 LoRA / QLoRA。
用法(在 Week1/ 根目录 + llama_factory 环境运行):
    conda activate llama_factory && cd ~/SenceTime/Week1
    python code/finetune/finetune.py                    # 默认 alpaca_zh_demo
    python code/finetune/finetune.py identity_penguin   # 指定语料

Mac(MPS,无 CUDA):QLORA=0(普通 LoRA)。
4090(CUDA):QLORA=4(4-bit QLoRA)+ MODEL 换 7B(任务要求)。
"""
import os
import sys
import subprocess

# ========= 可配置区 =========
MODEL   = "./models/Qwen2.5-1.5B-Instruct"   # 换 7B: ./models/Qwen2.5-7B-Instruct
DATASET = "alpaca_zh_demo"                     # 默认中文指令语料;命令行可覆盖
QLORA   = 0                                    # 0=普通LoRA(Mac);4=4-bit QLoRA(仅CUDA)
LF_DATA = "/Users/ruibosunsmacbook/SenceTime/LLaMA-Factory/data"  # 数据集目录
# ===========================

# 命令行覆盖语料:  python code/finetune.py identity_penguin
if len(sys.argv) > 1:
    DATASET = sys.argv[1]

output_dir = f"./saves/{os.path.basename(MODEL)}-{DATASET}-lora"

# 基础配置在 lf_lora.yaml;这里用命令行覆盖 model/dataset/output
args = [
    "llamafactory-cli", "train", "code/finetune/lf_lora.yaml",
    f"model_name_or_path={MODEL}",
    f"dataset={DATASET}",
    f"dataset_dir={LF_DATA}",
    f"output_dir={output_dir}",
]
if QLORA:  # 仅 CUDA 有效
    args.append(f"quantization_bit={QLORA}")

# MPS 不支持的算子自动回退 CPU,避免 Mac 上报错
env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")

print(f"模型={MODEL}  数据集={DATASET}  QLoRA={QLORA}  输出={output_dir}")
subprocess.run(args, env=env, check=True)
