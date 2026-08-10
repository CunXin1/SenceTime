"""
make_scorecard.py — Week3 Day14
生成盲测打分材料：匿名答卷 + 空白打分 CSV + 保密映射表。
Generate blind-test grading materials: anonymized answer sheets, an empty
scoring CSV, and a confidential identity mapping.

盲测设计 / Blind-test design:
    1. 模型名以固定种子（42）洗牌后映射为 Model-A/B/C…——打分人看不到真实身份，
       同一批模型每次生成的映射也一致（可复现）。
       Model names are shuffled with a fixed seed (42) into Model-A/B/C… —
       graders never see real identities, and the mapping is reproducible.
    2. 答卷按"题目 → 各模型回答"组织，方便横向对比打分。
       Answer sheets are grouped by question for side-by-side grading.
    3. 映射表单独成文件，打分完成前不发给打分人。
       The mapping lives in a separate file — do NOT share before grading ends.

输入 / Input : Week3/deliverables/eval_answers/answers_*.json（eval_harness.py 产物）
输出 / Output:
    Week3/deliverables/盲测答卷.md        匿名答卷（发给打分人）
    Week3/deliverables/盲测打分记录.csv    空白打分表（发给打分人，Excel 可直接开）
    Week3/deliverables/盲测映射表.md       真实身份映射（保密）
    Week3/deliverables/盲测映射.json       同上，机器可读（make_radar.py 消费）

用法 / Usage（仓库根目录 / from repo root）:
    # 默认对全部已生成答卷的模型出题；建议用 --models 只挑入围模型，
    # 控制打分工作量（16 模型 x 20 题 x 5 维 = 1600 个格子太多）。
    .venv/Scripts/python.exe Week3/code/make_scorecard.py --models qwen_base,qwen_r8_lr1e-4_ep3,...
"""

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANSWERS_DIR = ROOT / "Week3" / "deliverables" / "eval_answers"
QUESTIONS = ROOT / "Week3" / "data" / "eval_questions.json"
DELIV = ROOT / "Week3" / "deliverables"

# 5 个评分维度（与 人工评分卡.md 一致）/ the 5 dimensions (match the rubric)
DIMS = ["准确性", "完整性", "逻辑性", "安全性", "格式"]


def load_answers(models_filter: list) -> dict:
    """读取答卷 JSON，返回 {run_id: {question_id: answer}}。
    Load answer JSONs -> {run_id: {question_id: answer}}."""
    result = {}
    for p in sorted(ANSWERS_DIR.glob("answers_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        rid = data["run_id"]
        if models_filter and rid not in models_filter:
            continue
        result[rid] = {r["question_id"]: r["answer"] for r in data["records"]}
    return result


def anonymize(run_ids: list) -> dict:
    """固定种子洗牌 → Model-A/B/C…；映射可复现。
    Seeded shuffle -> Model-A/B/C…; reproducible mapping."""
    ids = sorted(run_ids)
    random.Random(42).shuffle(ids)
    return {rid: f"Model-{chr(ord('A') + i)}" for i, rid in enumerate(ids)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="",
                    help="逗号分隔的入围 run_id 列表（空=全部）/ comma-separated shortlist")
    args = ap.parse_args()
    models_filter = [m.strip() for m in args.models.split(",") if m.strip()]

    answers = load_answers(models_filter)
    if not answers:
        raise SystemExit("未找到答卷，请先跑 eval_harness.py / no answers found")
    mapping = anonymize(list(answers))
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    # 匿名代号排序，答卷与 CSV 都按此顺序 / stable anon order everywhere
    anon_order = sorted(answers, key=lambda r: mapping[r])

    # ---- 盲测答卷.md（发打分人 / for graders）----
    lines = ["# Week3 盲测答卷（匿名）", "",
             "> 打分方法见《人工评分卡.md》；参考要点在每题末尾。",
             f"> 共 {len(answers)} 个匿名模型 × {len(questions)} 题。", ""]
    for q in questions:
        lines += [f"## {q['id']}（{q['category']}）", "", f"**题目**：{q['question']}", ""]
        for rid in anon_order:
            ans = answers[rid].get(q["id"], "（该模型无此题答案 / missing）")
            lines += [f"### {mapping[rid]}", "", ans, ""]
        lines += [f"> **参考要点**：{q['reference']}", "", "---", ""]
    (DELIV / "盲测答卷.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- 盲测打分记录.csv（utf-8-sig：Excel 直接打开不乱码）----
    # utf-8-sig so Excel opens Chinese headers correctly.
    with open(DELIV / "盲测打分记录.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["匿名模型", "题目ID", "类别"] + DIMS + ["备注"])
        for rid in anon_order:
            for q in questions:
                w.writerow([mapping[rid], q["id"], q["category"],
                            "", "", "", "", "", ""])

    # ---- 映射表（保密 / CONFIDENTIAL）----
    (DELIV / "盲测映射.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    mlines = ["# 盲测映射表（保密——打分完成前勿发给打分人）", "",
              "| 匿名代号 | 真实模型 |", "|---|---|"]
    for rid in anon_order:
        mlines.append(f"| {mapping[rid]} | `{rid}` |")
    (DELIV / "盲测映射表.md").write_text("\n".join(mlines) + "\n", encoding="utf-8")

    print(f"[OK] {len(answers)} models -> 盲测答卷.md / 盲测打分记录.csv / 盲测映射表.md")
    for rid in anon_order:
        print(f"  {mapping[rid]} <- {rid}")


if __name__ == "__main__":
    main()
