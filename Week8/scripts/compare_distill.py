"""
compare_distill.py — Week8 Day42 / 任务书 42.3
对比蒸馏前后学生模型的 CEval 分数与推理速度，产出效果对比表。
Compares the student arms on CEval and inference speed; emits the Day42 table.

===========================================================================
★ 为什么是四组而不是两组
===========================================================================
    任务书写的是"对比蒸馏前后学生模型"，字面上只需要两组：
        A 蒸馏前（Qwen2.5-0.5B-Instruct 原始基座）
        C 蒸馏后（KD 训练过的 0.5B）
    但 A→C 的差值里混着**两样东西**：
        ① 在这批数据上又训了 2 轮 —— 纯 SFT 也能拿到的收益
        ② 教师软标签带来的额外信息 —— 真正属于"蒸馏"的收益
    只报 A→C，就是把 ① 的功劳记在蒸馏头上。所以必须加一组：
        B 纯 SFT 对照（alpha=0，同数据、同超参、同种子，教师不参与）
    **C − B 才是蒸馏的净效果。**
    再加上教师本身作为天花板参照：
        D 教师（Qwen2.5-3B-week4-dpo-merged）
    四组一起看，才能回答"蒸馏把学生推到了教师的百分之几"这个真问题。
    Without arm B, any gain is confounded with "two more epochs of SFT".

★ 三个指标各自回答什么
    · CEval（ppl-5shot，52 学科 1346 题）—— 知识与推理的**保真度**。
      任务书 42.3 点名要 CEval。用自带的 ceval_local.py，口径写在结果里。
    · 20 题集自动 5 维分 —— 生成质量的**代理指标**（对第 3 周 100 条人工打分
      回归，模型级 rho=0.90）。CEval 是选择题，测不出"写得好不好"。
    · 推理速度 tok/s —— 蒸馏的**动机本身**。如果 0.5B 跑得不比 3B 快多少，
      整件事就没有意义了。三个指标缺一不可：只看 CEval 会漏掉"更快"，
      只看速度会漏掉"更笨"。

★ 速度口径：单条贪心解码的端到端 tok/s，不是吞吐
    直接复用 20 题集生成时记录的 gen_seconds / new_tokens，
    即 batch=1、贪心、含 prefill 的**端到端**速率。这与第 7 周 vLLM 的
    并发吞吐（tok/s @ concurrency=16）不是同一个口径，**不能互相比较**。
    选这个口径是因为它对应"一个人用聊天框"的真实体感，
    而且四组用完全相同的题目与生成参数，横向可比。

用法 / Usage（仓库根目录）:
    # 全跑（四组 × CEval + 20 题）
    .venv/Scripts/python.exe Week8/scripts/compare_distill.py
    # 只跑学生三组，跳过 3B 教师（省时间）
    .venv/Scripts/python.exe Week8/scripts/compare_distill.py --skip-teacher
    # 冒烟：CEval 每科 2 题
    .venv/Scripts/python.exe Week8/scripts/compare_distill.py --ceval-limit 2 --skip-teacher

产物 / Output:
    Week8/deliverables/蒸馏效果对比表.md
    Week8/deliverables/distill_compare.json
    Week8/deliverables/ceval/<tag>.json       （各组的逐学科 CEval 明细）
    Week8/deliverables/eval_answers/<tag>.json（各组的 20 题答卷）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

DELIV = ROOT / "Week8" / "deliverables"
OUT_MD = DELIV / "蒸馏效果对比表.md"
OUT_JSON = DELIV / "distill_compare.json"

# 四个对照组。顺序即表格顺序：基座 → 纯 SFT → KD → 教师。
ARMS = [
    ("A_student_base", "models/Qwen2.5-0.5B-Instruct",
     "A · 学生基座", "Qwen2.5-0.5B-Instruct，未做任何训练"),
    ("B_student_sft", "models/Qwen2.5-0.5B-week8-sft",
     "B · 纯 SFT 对照", "alpha=0，同数据同超参同种子，教师不参与"),
    ("C_student_kd", "models/Qwen2.5-0.5B-week8-distill",
     "C · KD 蒸馏", "alpha=0.5，T=2.0，教师在线前向"),
    ("D_teacher", "models/Qwen2.5-3B-week4-dpo-merged",
     "D · 教师（3B）", "第 4 周 DPO 模型，作为天花板参照"),
]


def fmt(v, nd=2, dash="—"):
    if v is None:
        return dash
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ceval-limit", type=int, default=0,
                    help="每个学科最多评几题（0=全部 1346 题）")
    ap.add_argument("--skip-teacher", action="store_true",
                    help="跳过 3B 教师组（省约 15 分钟 GPU）")
    ap.add_argument("--skip-ceval", action="store_true")
    ap.add_argument("--skip-gen", action="store_true",
                    help="跳过 20 题生成（那样也就没有速度数据）")
    args = ap.parse_args()

    from auto_score import AutoScorer
    from ceval_local import run_ceval
    from step3_eval import ANSWER_DIR, generate_answers, load_questions

    arms = [a for a in ARMS if not (args.skip_teacher and a[0] == "D_teacher")]

    # 缺哪一组就明说，不静默跳过——少一组会让整张表的结论变味。
    missing = [(t, p) for t, p, _, _ in arms if not (ROOT / p).exists()]
    if missing:
        print("[FAIL] 以下模型目录不存在，无法组成完整对照：")
        for t, p in missing:
            print(f"        {t:<18} {p}")
        print("       B/C 两组由 distill_kd.py 产出：")
        print("         .venv/Scripts/python.exe Week8/scripts/distill_kd.py "
              "--config Week8/configs/student_sft_baseline.yaml")
        print("         .venv/Scripts/python.exe Week8/scripts/distill_kd.py "
              "--config Week8/configs/distill_kd.yaml")
        return 1

    scorer = AutoScorer()
    questions = load_questions()
    results: dict[str, dict] = {}

    for tag, rel, label, desc in arms:
        path = ROOT / rel
        print("=" * 78)
        print(f"[{tag}] {label} — {path.name}")
        print("=" * 78, flush=True)
        row: dict = {"tag": tag, "label": label, "desc": desc, "model": rel}

        # ---- CEval ----
        if not args.skip_ceval:
            r = run_ceval(path, tag, n_shot=5, limit=args.ceval_limit, verbose=True)
            row["ceval"] = {k: r[k] for k in
                            ("Average", "STEM", "Social Sciences", "Humanities",
                             "Other", "caliber", "n_questions", "seconds")}
            print(f"  → CEval Average {r['Average']}  ({r['n_questions']} 题, {r['seconds']}s)")

        # ---- 20 题生成 + 自动 5 维分 + 速度 ----
        if not args.skip_gen:
            recs = generate_answers(path, questions, min_free_gb=6.0)
            ANSWER_DIR.mkdir(parents=True, exist_ok=True)
            (ANSWER_DIR / f"{tag}.json").write_text(
                json.dumps({"run_id": tag, "model": str(path), "records": recs},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            res = scorer.score_records(recs)
            gen_s = sum(x["gen_seconds"] for x in recs)
            n_tok = sum(x["new_tokens"] for x in recs)
            row["auto5"] = res["dims"]
            row["speed"] = {"gen_seconds": round(gen_s, 1), "new_tokens": n_tok,
                            "tok_per_s": round(n_tok / gen_s, 1) if gen_s else None}
            print(f"  → 自动 5 维总分 {res['dims']['total']}   "
                  f"{row['speed']['tok_per_s']} tok/s")

        results[tag] = row

    OUT_JSON.write_text(json.dumps(
        {"generated_at": datetime.now().isoformat(timespec="seconds"),
         "arms": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # 渲染对比表
    # ------------------------------------------------------------------
    def g(tag, *ks, default=None):
        d = results.get(tag)
        for k in ks:
            if d is None:
                return default
            d = d.get(k)
        return d if d is not None else default

    L = ["# Week8 Day42 蒸馏效果对比表",
         "",
         f"> 由 `Week8/scripts/compare_distill.py` 自动生成 · {datetime.now():%Y-%m-%d %H:%M}",
         "> 所有数字均为实测，未做任何插值或估算。",
         "",
         "## 一、四个对照组",
         "",
         "| 组 | 模型 | 说明 |", "|---|---|---|"]
    for tag, rel, label, desc in arms:
        L.append(f"| **{label}** | `{rel}` | {desc} |")

    L += ["",
          "> ★ **B 组存在的意义**：A→C 的差值里混着「又训了 2 轮」和「教师软标签」",
          "> 两样东西。只报 A→C 等于把纯 SFT 的功劳记在蒸馏头上。",
          "> **C − B 才是蒸馏的净效果。**",
          "",
          "## 二、CEval（ppl-5shot 口径，官方 val 划分）",
          "",
          "| 组 | Average | STEM | 社科 | 人文 | 其他 | 题数 | 耗时 |",
          "|---|---|---|---|---|---|---|---|"]
    for tag, _, label, _ in arms:
        L.append(f"| {label} | **{fmt(g(tag,'ceval','Average'))}** | "
                 f"{fmt(g(tag,'ceval','STEM'))} | {fmt(g(tag,'ceval','Social Sciences'))} | "
                 f"{fmt(g(tag,'ceval','Humanities'))} | {fmt(g(tag,'ceval','Other'))} | "
                 f"{fmt(g(tag,'ceval','n_questions'),0)} | {fmt(g(tag,'ceval','seconds'),1)}s |")

    L += ["",
          "> ★ **口径说明**：本表用 ppl 口径（比较 A/B/C/D 四个选项 token 的 logit，",
          "> 每题一次前向），**不是** OpenCompass 的 `ceval_gen` 口径（自由生成后正则抽取）。",
          "> ppl 口径对小模型更宽容（不惩罚「不会按格式作答」），分数系统性偏高，",
          "> **不能直接与论文里标 `ceval_gen` 的数字比较**。四组内部横向可比。",
          "",
          "## 三、20 题自定义集（自动 5 维打分）",
          "",
          "| 组 | 准确性 | 完整性 | 逻辑性 | 安全性 | 格式 | 加权总分 |",
          "|---|---|---|---|---|---|---|"]
    for tag, _, label, _ in arms:
        L.append(f"| {label} | {fmt(g(tag,'auto5','accuracy'),3)} | "
                 f"{fmt(g(tag,'auto5','completeness'),3)} | {fmt(g(tag,'auto5','logic'),3)} | "
                 f"{fmt(g(tag,'auto5','safety'),3)} | {fmt(g(tag,'auto5','format'),3)} | "
                 f"**{fmt(g(tag,'auto5','total'),3)}** |")

    L += ["",
          "## 四、推理速度（batch=1，贪心，含 prefill，20 题端到端）",
          "",
          "| 组 | 总生成 token | 总耗时 | tok/s | 相对教师 |",
          "|---|---|---|---|---|"]
    d_tps = g("D_teacher", "speed", "tok_per_s")
    for tag, _, label, _ in arms:
        tps = g(tag, "speed", "tok_per_s")
        rel = f"{tps / d_tps:.2f}×" if (tps and d_tps) else "—"
        L.append(f"| {label} | {fmt(g(tag,'speed','new_tokens'),0)} | "
                 f"{fmt(g(tag,'speed','gen_seconds'),1)}s | **{fmt(tps,1)}** | {rel} |")

    L += ["",
          "> ★ 这是 **batch=1 单条贪心**的端到端速率，与第 7 周 vLLM 的并发吞吐",
          "> （tok/s @ concurrency=16）不是同一个口径，不能互比。选这个口径是因为",
          "> 它对应「一个人用聊天框」的真实体感，且四组用完全相同的题目与生成参数。",
          "",
          "## 五、蒸馏净效果（C − B）",
          ""]

    b_ce, c_ce = g("B_student_sft", "ceval", "Average"), g("C_student_kd", "ceval", "Average")
    a_ce = g("A_student_base", "ceval", "Average")
    b_a5, c_a5 = g("B_student_sft", "auto5", "total"), g("C_student_kd", "auto5", "total")
    a_a5 = g("A_student_base", "auto5", "total")
    L += ["| 指标 | A 基座 | B 纯SFT | C 蒸馏 | **C − B（蒸馏净效果）** | C − A（含SFT收益） |",
          "|---|---|---|---|---|---|"]
    if None not in (a_ce, b_ce, c_ce):
        L.append(f"| CEval Average | {a_ce:.2f} | {b_ce:.2f} | {c_ce:.2f} | "
                 f"**{c_ce - b_ce:+.2f}** | {c_ce - a_ce:+.2f} |")
    if None not in (a_a5, b_a5, c_a5):
        L.append(f"| 自动 5 维总分 | {a_a5:.3f} | {b_a5:.3f} | {c_a5:.3f} | "
                 f"**{c_a5 - b_a5:+.3f}** | {c_a5 - a_a5:+.3f} |")

    L += ["",
          "> 结论分析见 `Week8/docs/Day42_知识蒸馏.md` 第四节。",
          ""]

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n[ok] {OUT_MD}")
    print(f"[ok] {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
