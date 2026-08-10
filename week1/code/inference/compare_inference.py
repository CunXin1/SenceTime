"""四模型横向对比:同一问题,4 个模型的回答并列输出。

逐个加载模型(省 16GB 内存),跑完 3 个问题就释放,最后按【问题】分组写 markdown。
输出:deliverables/推理输出/4模型推理输出.md

用法(在 Week1/ 根目录 + llm_exp 环境运行):
    conda activate llm_exp && cd ~/SenceTime/Week1
    python code/inference/compare_inference.py
"""
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")

PROMPTS = [
    ("代码生成", "用 Python 写一个判断字符串是否回文的函数,并给测试例子。"),
    ("逻辑推理", "小明比小红大3岁,小红比小刚大2岁,小刚10岁。小明几岁?一步步推理。"),
    ("角色扮演", "你是中学物理老师,用生活例子给初中生解释牛顿第一定律。"),
]

MODELS = [
    ("Qwen2.5-1.5B", "./models/Qwen2.5-1.5B-Instruct", "hf"),
    ("Qwen3-4B",     "./models/Qwen3-4B",              "hf"),
    ("Llama-3.2-3B", "./models/Llama-3.2-3B-Instruct", "hf"),
    ("Gemma-4-E2B",  "./models/gemma-4-E2B-it",        "gemma"),
]

SYS = "You are a helpful assistant."


def free(model):
    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()


def run_hf(path):  # Qwen / Llama
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map=DEVICE).eval()
    out = []
    for _, p in PROMPTS:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": p}]
        try:  # enable_thinking=False 关 Qwen3 思考;其他模型走 except
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            g = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.8)
        out.append(tok.decode(g[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())
    free(model)
    return out


def run_gemma(path):  # Gemma-4 多模态,用 AutoProcessor
    proc = AutoProcessor.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map=DEVICE).eval()
    out = []
    for _, p in PROMPTS:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": p}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = proc(text=text, return_tensors="pt").to(DEVICE)
        ilen = inputs["input_ids"].shape[-1]
        with torch.no_grad():  # Gemma 官方推荐采样参数
            g = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=1.0, top_p=0.95, top_k=64)
        resp = proc.decode(g[0][ilen:], skip_special_tokens=False)
        try:
            parsed = proc.parse_response(resp)
            out.append((parsed["content"] if isinstance(parsed, dict) else parsed).strip())
        except Exception:
            out.append(proc.decode(g[0][ilen:], skip_special_tokens=True).strip())
    free(model)
    return out


results = {}
for name, path, kind in MODELS:
    print(f"[跑] {name}", flush=True)
    results[name] = run_gemma(path) if kind == "gemma" else run_hf(path)

# 按【问题】分组写 markdown
lines = ["# 4 模型横向对比(同问题并列)", "",
         "> 同一问题下并列 4 个模型的回答,便于横向比较。采样:Qwen/Llama temp=0.7;Gemma 用官方 temp=1.0。", ""]
for qi, (domain, prompt) in enumerate(PROMPTS):
    lines += [f"---", "", f"# 问题 {qi + 1}【{domain}】", "", f"> {prompt}", ""]
    for name, _, _ in MODELS:
        lines += [f"## {name}", "", results[name][qi], ""]

with open("deliverables/推理输出/4模型推理输出.md", "w") as f:
    f.write("\n".join(lines))
print("DONE 已写入 deliverables/推理输出/4模型推理输出.md")
