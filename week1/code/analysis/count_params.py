"""统计模型参数量,理解"参数堆在哪" (Day 3)。
用法(在 Week1/ 根目录 + llm_exp 环境运行):
    conda activate llm_exp && cd ~/SenceTime/Week1
    python code/analysis/count_params.py [模型路径]
"""
import sys
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM

path = sys.argv[1] if len(sys.argv) > 1 else "./models/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16)

# 总参数量:遍历所有权重张量,累加元素个数(numel = 参数个数)
total = sum(p.numel() for p in model.parameters())

# 按模块分类累加:named_parameters() 给出 (名字, 张量)
buckets = defaultdict(int)
for name, p in model.named_parameters():
    if   "embed" in name:     key = "embedding 词表"
    elif "self_attn" in name: key = "attention 注意力"
    elif "mlp" in name:       key = "mlp 前馈"
    elif "norm" in name:      key = "norm 归一化"
    else:                     key = "其他"
    buckets[key] += p.numel()

print(f"模型: {path}")
print(f"总参数量: {total:,}  ({total/1e9:.3f}B)\n")
print(f"{'模块':<16}{'参数量':>15}{'占比':>8}")
print("-" * 40)
for key, n in sorted(buckets.items(), key=lambda kv: -kv[1]):
    print(f"{key:<16}{n:>15,}{n/total*100:>7.1f}%")
