"""Day23 交付:批量图文推理 -> 图文推理结果表.csv + 原始 JSON

对每张图跑 5 类问题(描述/提取文字/解释图表/评价美学/推理隐含信息),两个模型各跑一遍。
CSV 末尾预留 3 个空列 score / hallucination / note —— **必须人工填写**,
因为 ❶「能完成图文推理」的验收看的不是有没有输出,而是输出对不对。

用法:
    .venv-vlm\\Scripts\\python.exe Week5/code/vlm_infer.py               # 两个模型都跑
    .venv-vlm\\Scripts\\python.exe Week5/code/vlm_infer.py --model qwen
    .venv-vlm\\Scripts\\python.exe Week5/code/vlm_infer.py --core-only   # 只跑 25 条核心
    .venv-vlm\\Scripts\\python.exe Week5/code/vlm_infer.py --resume      # 续跑,跳过已有结果
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import ROOT, SPECS, generate, load_vlm  # noqa: E402

IMG_DIR = ROOT / "Week5" / "data" / "images"
QUESTIONS = ROOT / "Week5" / "data" / "questions.json"
DELIV = ROOT / "Week5" / "deliverables"
RAW_JSON = DELIV / "day23_raw.json"
OUT_CSV = DELIV / "图文推理结果表.csv"

CSV_FIELDS = [
    "qid", "model", "image", "category_cn", "is_core", "question", "answer",
    "image_tokens", "prompt_tokens", "new_tokens", "latency_s", "peak_mem_gb",
    "score", "hallucination", "note",
]


def load_questions(core_only: bool) -> list[dict]:
    rows = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows = [r for r in rows if (IMG_DIR / r["image"]).exists()]
    if core_only:
        rows = [r for r in rows if r["is_core"]]
    return rows


def run_model(key: str, questions: list[dict], done: set[tuple[str, str]]) -> list[dict]:
    todo = [q for q in questions if (key, q["qid"]) not in done]
    if not todo:
        print(f"[{key}] 全部已完成,跳过加载")
        return []

    print(f"\n{'=' * 78}\n[{key}] {SPECS[key].display} —— 待跑 {len(todo)}/{len(questions)} 条\n{'=' * 78}",
          flush=True)
    vlm = load_vlm(key, attn_impl="sdpa")
    print(f"[加载完成] {vlm.load_seconds:.1f}s\n", flush=True)

    out = []
    t_start = time.perf_counter()   # 单调时钟:本机系统时钟会跳，挂钟算出的 ETA 会离谱
    for i, q in enumerate(todo, 1):
        r = generate(vlm, IMG_DIR / q["image"], q["question"])
        out.append({
            "qid": q["qid"], "model": key, "image": q["image"],
            "category_cn": q["category_cn"], "is_core": q["is_core"],
            "question": q["question"], "answer": r.text,
            "image_tokens": r.image_tokens, "prompt_tokens": r.prompt_tokens,
            "new_tokens": r.new_tokens, "latency_s": round(r.latency_s, 2),
            "peak_mem_gb": round(r.peak_mem_gb, 2),
            "score": "", "hallucination": "", "note": "",
        })
        eta = (time.perf_counter() - t_start) / i * (len(todo) - i)
        print(f"[{i}/{len(todo)}] {q['qid']:<20} {q['category_cn']:<7} "
              f"imgtok={r.image_tokens:<5} {r.latency_s:>5.1f}s  ETA {eta / 60:.1f}min")
        print(f"    → {r.text[:180].replace(chr(10), ' ')}\n", flush=True)

    del vlm
    torch.cuda.empty_cache()
    return out


def write_csv(rows: list[dict]) -> None:
    order = {k: i for i, k in enumerate(SPECS)}
    rows = sorted(rows, key=lambda r: (r["image"], order.get(r["model"], 9), r["qid"]))
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"\n[写出] {OUT_CSV}  共 {len(rows)} 行")
    print("  ⚠️  score(1-5) / hallucination(0|1) / note 三列需人工填写,Day23 分析和 ❶ 验收靠它")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SPECS), help="默认两个都跑")
    ap.add_argument("--core-only", action="store_true", help="只跑 5图×5类=25 条核心")
    ap.add_argument("--resume", action="store_true", help="跳过 day23_raw.json 里已有的结果")
    args = ap.parse_args()

    questions = load_questions(args.core_only)
    if not questions:
        sys.exit("[中止] 没有可跑的问题:检查图片是否就绪(prepare_images.py --check)")

    missing = sorted({r["image"] for r in json.loads(QUESTIONS.read_text(encoding="utf-8"))
                      if not (IMG_DIR / r["image"]).exists()})
    if missing:
        print(f"⚠️  以下图片缺失,相关问题已跳过:{missing}")
        print("   见 Week5/data/README_图片素材.md\n")

    existing: list[dict] = []
    if args.resume and RAW_JSON.exists():
        existing = json.loads(RAW_JSON.read_text(encoding="utf-8"))
    done = {(r["model"], r["qid"]) for r in existing}

    keys = [args.model] if args.model else list(SPECS)
    all_rows = list(existing)
    for k in keys:
        all_rows += run_model(k, questions, done)
        RAW_JSON.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    write_csv(all_rows)

    # 汇总:每个模型的平均延迟 / 平均视觉 token
    print("\n=== 概览 ===")
    for k in keys:
        rs = [r for r in all_rows if r["model"] == k]
        if rs:
            print(f"  {SPECS[k].display:<26} {len(rs):>3} 条  "
                  f"平均视觉token {sum(r['image_tokens'] for r in rs) / len(rs):>7.0f}  "
                  f"平均延迟 {sum(r['latency_s'] for r in rs) / len(rs):>5.1f}s  "
                  f"峰值显存 {max(r['peak_mem_gb'] for r in rs):>5.2f}GB")
