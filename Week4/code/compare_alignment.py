"""
compare_alignment.py — Week4 Day20
对齐效果对比：5 个业务 Prompt 上 SFT-only vs SFT+DPO 的并排答卷 + 空白主观评分表。
Alignment comparison: SFT-only vs SFT+DPO side-by-side on 5 business prompts,
plus a blank subjective scoring sheet.

每个 Prompt 对应一种偏好类型（事实/完整/有用/格式/安全边界），第 5 题是合法但敏感的
话题，专门检验 DPO 后是否过度拒绝（over-refusal）。评分沿用 Week3 人工评分卡的 5 维权重。

复用 Week3 eval_harness.py 的热挂载模式：policy 加载一次，DPO adapter 动态挂载。

输出 / Output:
    Week4/deliverables/对齐效果对比表.md    并排答卷
    Week4/deliverables/对齐主观评分.csv      空白 5 维评分表（utf-8-sig）

用法 / Usage（仓库根目录 / from repo root）:
    .venv/Scripts/python.exe Week4/code/compare_alignment.py
    .venv/Scripts/python.exe Week4/code/compare_alignment.py --dpo-run qwen_dpo_beta0.5_lr5e-6
"""

import argparse
import csv
import gc
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "Week4" / "data" / "business_prompts.json"
DELIV = ROOT / "Week4" / "deliverables"
OUT_MD = DELIV / "对齐效果对比表.md"
CSV_PATH = DELIV / "对齐主观评分.csv"

POLICY = ROOT / "models" / "Qwen2.5-3B-week3-best-merged"
MAX_NEW_TOKENS = 512
# 与 Week3 人工评分卡.md 一致 / same 5 dimensions and weights as Week3
DIMS = ["准确性", "完整性", "逻辑性", "安全性", "格式"]


def load_prompts() -> list:
    return json.loads(PROMPTS.read_text(encoding="utf-8"))["prompts"]


def generate(model, tok, prompts: list, tag: str) -> list:
    import torch
    records = []
    for i, q in enumerate(prompts, 1):
        text = tok.apply_chat_template(
            [{"role": "user", "content": q["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        ans = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        records.append({"id": q["id"], "category": q["category"], "answer": ans})
        print(f"  [{tag}] {i}/{len(prompts)} {q['id']}  {round(time.time()-t0,1)}s")
    return records


def run(dpo_run: str) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_prompts()
    tok = AutoTokenizer.from_pretrained(POLICY, trust_remote_code=True)
    print(f"[load] policy {POLICY.name}")
    base = AutoModelForCausalLM.from_pretrained(
        POLICY, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    base.eval()

    out = {"sft_only": generate(base, tok, prompts, "sft_only")}
    adapter = ROOT / "saves" / "week4" / "qwen" / dpo_run
    if not (adapter / "adapter_model.safetensors").exists():
        raise SystemExit(f"DPO adapter 不存在：{adapter}（训练未完成？）")
    peft_model = PeftModel.from_pretrained(base, adapter)
    peft_model.eval()
    out[dpo_run] = generate(peft_model, tok, prompts, dpo_run)

    del peft_model, base, tok
    gc.collect()
    torch.cuda.empty_cache()
    return out


def write_outputs(results: dict, dpo_run: str) -> None:
    prompts = load_prompts()
    models = ["sft_only", dpo_run]

    lines = ["# Week4 Day20 交付：对齐效果对比表（SFT-only vs SFT+DPO）", "",
             f"> 5 个通用中文助手 Prompt。对比 `sft_only` 与 `{dpo_run}`（最优 DPO）。",
             "> 每题对应一种偏好类型；biz-05 是合法敏感话题，检验是否过度拒绝。",
             "> 主观评分请填 `对齐主观评分.csv`（5 维 1–5，权重 30/25/20/15/10）。", ""]
    for q in prompts:
        lines += [f"## {q['id']}（{q['category']}）", "",
                  f"**Prompt**：{q['prompt']}", "",
                  f"**考察点**：{q['focus']}", ""]
        for m in models:
            rec = next((r for r in results[m] if r["id"] == q["id"]), None)
            lines += [f"### [{m}]", "", rec["answer"] if rec else "（无答案）", ""]
        lines += ["---", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["模型", "题目ID", "类别"] + DIMS + ["备注"])
        for m in models:
            for q in prompts:
                w.writerow([m, q["id"], q["category"]] + [""] * len(DIMS) + [""])
    print(f"[OK] 对比答卷 -> {OUT_MD.relative_to(ROOT)}")
    print(f"[OK] 空白评分表 -> {CSV_PATH.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpo-run", default="qwen_dpo_beta0.1_lr5e-6",
                    help="要对比的 DPO run_id（默认基线组）")
    args = ap.parse_args()
    results = run(args.dpo_run)
    write_outputs(results, args.dpo_run)


if __name__ == "__main__":
    main()
