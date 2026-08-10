"""微调前后对比:基座模型 vs LoRA 微调后,问"你是谁"看身份变化。

用法(在 Week1/ 根目录 + llama_factory 环境运行):
    conda activate llama_factory && cd ~/SenceTime/Week1
    python code/finetune/compare_finetune.py
原理:基座不变,PeftModel 把 adapter 贴上去 → 对比同样问题的回答。
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE    = "./models/Qwen2.5-1.5B-Instruct"
ADAPTER = "./saves/Qwen2.5-1.5B-Instruct-identity_penguin-lora"
DEVICE  = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")

QUESTIONS = ["你是谁?", "Who are you?", "你叫什么名字?是谁开发的?"]

tok = AutoTokenizer.from_pretrained(BASE)


def ask(model, q):
    text = tok.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():  # 贪心解码,结果确定、可复现
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# 先用基座回答(此时还没挂 adapter)
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map=DEVICE).eval()
print("=" * 60, "\n微调前(基座 Qwen2.5-1.5B)\n", "=" * 60, sep="")
before = {q: ask(base, q) for q in QUESTIONS}
for q, a in before.items():
    print(f"Q: {q}\nA: {a}\n")

# 把 identity_penguin 的 LoRA adapter 贴到基座上
ft = PeftModel.from_pretrained(base, ADAPTER).eval()
print("=" * 60, "\n微调后(+ identity_penguin LoRA)\n", "=" * 60, sep="")
for q in QUESTIONS:
    print(f"Q: {q}\nA: {ask(ft, q)}\n")
