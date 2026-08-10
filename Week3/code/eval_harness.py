"""
eval_harness.py — Week3 Day14
用 20 道固定高难度题批量测试所有模型（2 个基座 + 14 个 LoRA adapter）。
Batch-test every model (2 bases + 14 LoRA adapters) on the 20 fixed hard
questions.

关键设计 / Key design:
    1. 基座只加载一次，adapter 用 PEFT 动态挂载/卸载——不做 14 次合并导出，
       省约 80GB 磁盘与大量时间。
       Load each base once and hot-swap LoRA adapters via PEFT — no 14x
       merge-exports (saves ~80GB disk and hours).
    2. 贪心解码（do_sample=False）：同一模型同一题的输出完全可复现。
       Greedy decoding: fully reproducible outputs.
    3. 所有模型用同一份题目、同一顺序、同一生成参数，保证公平对比。
       Identical questions, order and generation params for every model.
    4. 显存管理沿用 Week2/code/compare_finetune.py 的模式：换基座前
       gc + torch.cuda.empty_cache()。
       VRAM management follows Week2's compare_finetune.py pattern.

输出 / Output:
    Week3/deliverables/eval_answers/answers_{run_id}.json   逐题答案（机器可读）
    Week3/deliverables/eval_answers/{run_id}.md             逐题答案（人读）

用法 / Usage（仓库根目录 / from repo root）:
    .venv/Scripts/python.exe Week3/code/eval_harness.py                     # 全部 16 个模型
    .venv/Scripts/python.exe Week3/code/eval_harness.py --only qwen_base    # 按子串过滤
    .venv/Scripts/python.exe Week3/code/eval_harness.py --max-questions 2   # 冒烟测试
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "Week3" / "configs" / "exp" / "experiments.json"
QUESTIONS = ROOT / "Week3" / "data" / "eval_questions.json"
OUT_DIR = ROOT / "Week3" / "deliverables" / "eval_answers"

# 两个基座的本地路径 / local paths of the two base models
BASES = {
    "qwen": ROOT / "models" / "Qwen2.5-3B-Instruct",
    "llama": ROOT / "models" / "Llama-3.2-3B-Instruct",
}

# 512 在 20 题上足够写完推理/代码，且单模型作答可控制在 ~5 分钟内；
# 所有模型同一上限，截断惩罚（若有）对各模型公平。
# 512 tokens suffice for these questions and keep each model under ~5 min;
# the same cap applies to every model, so any truncation penalty is fair.
MAX_NEW_TOKENS = 512


def load_base(family: str):
    """加载一个基座模型 + tokenizer（bf16, GPU）。
    Load one base model + tokenizer (bf16 on GPU)."""
    path = BASES[family]
    print(f"[load] base {family}: {path.name}")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    model.eval()
    return model, tok


def generate_answers(model, tok, questions: list, tag: str) -> list:
    """对一个模型跑完整题集，返回逐题记录。
    Run the full question set on one model; return per-question records."""
    records = []
    for i, q in enumerate(questions, 1):
        # 用各自的 chat template 组织对话（Qwen=ChatML / Llama=llama3），
        # 与训练/推理时的模板保持一致。
        # Use the model's own chat template (must match training).
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": q["question"]}],
            tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,                      # 贪心：可复现 / greedy: reproducible
                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        answer = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
        dt = round(time.time() - t0, 1)
        records.append({"question_id": q["id"], "category": q["category"],
                        "answer": answer, "gen_seconds": dt,
                        "new_tokens": int(out.shape[1] - inputs["input_ids"].shape[1])})
        print(f"  [{tag}] {i}/{len(questions)} {q['id']}  {dt}s")
    return records


def save_run(run_id: str, records: list) -> None:
    """保存 JSON（供 make_scorecard.py 消费）+ Markdown（人读）。
    Save JSON (consumed by make_scorecard.py) + Markdown (human-readable)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"answers_{run_id}.json").write_text(
        json.dumps({"run_id": run_id, "records": records},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    qmap = {q["id"]: q for q in json.loads(
        QUESTIONS.read_text(encoding="utf-8"))["questions"]}
    lines = [f"# 答卷：{run_id}", ""]
    for r in records:
        lines += [f"## {r['question_id']}（{r['category']}）", "",
                  f"**题目**：{qmap[r['question_id']]['question']}", "",
                  "**模型回答**：", "", r["answer"], "", "---", ""]
    (OUT_DIR / f"{run_id}.md").write_text("\n".join(lines), encoding="utf-8")


def free_vram(*objs) -> None:
    """释放显存（Week2 compare_finetune.py 同款模式）。
    Release VRAM (same pattern as Week2's compare_finetune.py)."""
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="只测 run_id 含此子串的模型 / substring filter")
    ap.add_argument("--max-questions", type=int, default=0,
                    help="只取前 N 题（冒烟用）/ first N questions (smoke)")
    ap.add_argument("--skip-done", action="store_true",
                    help="跳过已有 answers_*.json 的模型 / skip finished models")
    args = ap.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    if args.max_questions:
        questions = questions[:args.max_questions]
    exps = json.loads(MANIFEST.read_text(encoding="utf-8"))

    # 待测清单：每个家族 = 1 个基座 + 该家族全部 adapter。
    # Test roster: per family = 1 base + all of its adapters.
    for family in BASES:
        adapters = [e for e in exps if e["model"] == family]
        # 该家族一个 adapter 都没训（如只训 Qwen 时的 Llama）就整体跳过——
        # 基座单独评没有对比意义。 Skip a family entirely when none of its
        # adapters were trained (e.g. Llama in Qwen-only mode).
        if not any((ROOT / e["output_dir"] / "adapter_model.safetensors").exists()
                   for e in adapters):
            print(f"[skip] family {family}: 无已训练 adapter / no trained adapters")
            continue
        roster = [f"{family}_base"] + [e["run_id"] for e in adapters]
        roster = [r for r in roster if args.only in r]
        if args.skip_done:
            roster = [r for r in roster
                      if not (OUT_DIR / f"answers_{r}.json").exists()]
        if not roster:
            continue

        base, tok = load_base(family)
        for run_id in roster:
            if run_id.endswith("_base"):
                save_run(run_id, generate_answers(base, tok, questions, run_id))
                continue
            adapter_dir = ROOT / next(e["output_dir"] for e in adapters
                                      if e["run_id"] == run_id)
            if not (adapter_dir / "adapter_model.safetensors").exists():
                print(f"[skip] {run_id}: adapter 不存在（训练未完成？）/ adapter missing")
                continue
            # 挂载 adapter → 生成 → 卸载还原基座（unload 不合并权重，只摘掉旁路）。
            # Attach adapter -> generate -> unload back to the clean base
            # (unload removes LoRA modules without merging).
            peft_model = PeftModel.from_pretrained(base, adapter_dir)
            peft_model.eval()
            save_run(run_id, generate_answers(peft_model, tok, questions, run_id))
            base = peft_model.unload()
            del peft_model
            torch.cuda.empty_cache()
        free_vram(base, tok)  # 换家族前清空显存 / clear VRAM before next family

    print(f"\n[done] answers -> {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
