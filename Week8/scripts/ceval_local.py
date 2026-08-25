"""
ceval_local.py — Week8 Day41/42
一个自给自足的 CEval 评测器：直接读官方 `ceval/ceval-exam` 数据集，
5-shot、逐题单次前向、比较 A/B/C/D 四个候选 token 的 logit。
A self-contained CEval evaluator: official dataset, 5-shot, single forward
per question, argmax over the A/B/C/D option-token logits.

===========================================================================
★ 为什么要自己写，而不是继续等 OpenCompass
===========================================================================
    第 3 周 Day15 起，OpenCompass 就没在这台 Windows 机器上装成功过；
    `Week3/deliverables/OpenCompass评测分数表.md` 至今整张表都是 ⏳。
    LLaMA-Factory 自带的评测器本来是第二选择，但本地这个版本（0.9.6.dev0 @ 76a0391）
    上游**已经把 `evaluation/` 数据加载目录整个删掉了**——实测：

        git ls-files | grep '^evaluation'      → 空
        cached_file('evaluation/ceval', 'mapping.json')
          → OSError: evaluation/ceval is not a local folder and is not a
            valid model identifier listed on 'https://huggingface.co/models'

    也就是说前两级后端在本机都是死路。但真正卡住的其实**不是评测方法，而是数据**——
    而数据是拿得到的：`load_dataset('ceval/ceval-exam', name=<subject>)` 一行就下来了
    （52 个学科，val 划分共 1346 题，几 MB）。
    评测逻辑本身只有一百来行：拼 5-shot prompt、前向一次、比四个 logit。
    与其让交付文档里留 52 个 ⏳，不如把这一百行写掉。
    The blocker was never the method, it was the data — and the data is one
    `load_dataset` call away.

★ 口径必须写清楚，否则这个分数不能和别人的比
    CEval 有两种主流评测口径，分数不可直接互比：
      · **ppl / likelihood 口径**（本脚本）：把题目 + 选项拼进 prompt，让模型在
        「答案：」后面输出一个字母，取 A/B/C/D 四个 token 的 logit 最大者。
        每题 **1 次前向**，无采样，完全确定。
      · **gen 口径**（OpenCompass 的 `ceval_gen`）：让模型自由生成一段话，
        再用正则从里面抠出选项字母。每题要生成几十~几百 token，
        且抠不出来时算错——对不爱按格式作答的小模型很吃亏。
    本脚本用 ppl 口径，理由是 ①便宜（0.5B 上 1346 题几分钟）②确定（回归护栏需要）
    ③对小模型公平（不惩罚"不会按格式答"）。
    **代价：本脚本的分数偏高于 gen 口径，不能直接与论文里标 `ceval_gen` 的数字比较。**
    所有输出里都会带上 `caliber: ppl-5shot` 这个字段，不让口径丢失。
    Report the caliber with the score, always.

★ 5-shot 的示例从哪来
    每个学科的 `dev` 划分正好是官方给的 5 道带解析的示例题（每科恰好 5 条），
    这就是 CEval 设计好的 few-shot 池，不是我们从测试集里挖的。
    用 `val` 划分作为被测题（1346 题，带答案）；`test` 划分官方不公开答案，
    本地评不了，所以不用。
    dev = official 5-shot exemplars; val = the graded split.

★ 为什么比 logit 而不是解码一个字符
    解码要处理 tokenizer 把 " A" 和 "A" 编成不同 id、模型先输出空格/换行、
    输出 "选A" 这类前缀等一堆边界情况。直接取四个候选 token 在**同一个位置**上的
    logit 做 argmax，绕开全部这些问题，而且天然是确定性的。
    候选 token 的选取见 `_option_token_ids()`：对每个字母取
    tokenizer 编码出的**首个 token id**，并断言四个 id 互不相同——
    如果某个 tokenizer 把 A/B/C/D 编到同一个 id 上，这套方法就不成立，
    那时应当报错而不是给出一个悄悄错掉的分数。

用法 / Usage（仓库根目录）:
    .venv/Scripts/python.exe Week8/scripts/ceval_local.py \
        --model models/Qwen2.5-0.5B-Instruct --tag student_base
    # 冒烟：每科只评 2 题
    .venv/Scripts/python.exe Week8/scripts/ceval_local.py --model ... --tag x --limit 2
    # 只评部分学科
    .venv/Scripts/python.exe Week8/scripts/ceval_local.py --model ... --tag x \
        --subjects computer_network operating_system

产物 / Output:
    Week8/deliverables/ceval/<tag>.json    总分 + 四大类 + 逐学科
被 step3_eval.py 作为库导入：`from ceval_local import run_ceval`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Week8" / "deliverables" / "ceval"

CHOICES = ["A", "B", "C", "D"]

# ---------------------------------------------------------------------------
# 学科 → 四大类。取自 CEval 官方 subject_mapping（论文 Table 1 的分类）。
# 汇总分 "Average" 是**按题**平均（micro），不是先按学科算再平均（macro）——
# 各学科题数从 6 到 100+ 不等，macro 会让只有 6 题的学科和有 100 题的学科等权，
# 一两道题的抖动就能挪动总分。官方与 OpenCompass 都用 micro。
# Micro-average over questions, not macro over subjects.
# ---------------------------------------------------------------------------
SUBJECT_CATEGORY: dict[str, str] = {
    # ---- STEM ----
    "advanced_mathematics": "STEM", "college_chemistry": "STEM",
    "college_physics": "STEM", "college_programming": "STEM",
    "computer_architecture": "STEM", "computer_network": "STEM",
    "discrete_mathematics": "STEM", "electrical_engineer": "STEM",
    "high_school_biology": "STEM", "high_school_chemistry": "STEM",
    "high_school_mathematics": "STEM", "high_school_physics": "STEM",
    "metrology_engineer": "STEM", "middle_school_biology": "STEM",
    "middle_school_chemistry": "STEM", "middle_school_mathematics": "STEM",
    "middle_school_physics": "STEM", "operating_system": "STEM",
    "probability_and_statistics": "STEM", "veterinary_medicine": "STEM",
    # ---- Social Sciences ----
    "business_administration": "Social Sciences", "college_economics": "Social Sciences",
    "education_science": "Social Sciences", "high_school_geography": "Social Sciences",
    "high_school_politics": "Social Sciences", "mao_zedong_thought": "Social Sciences",
    "marxism": "Social Sciences", "middle_school_geography": "Social Sciences",
    "middle_school_politics": "Social Sciences", "teacher_qualification": "Social Sciences",
    # ---- Humanities ----
    "art_studies": "Humanities", "chinese_language_and_literature": "Humanities",
    "high_school_chinese": "Humanities", "high_school_history": "Humanities",
    "ideological_and_moral_cultivation": "Humanities", "law": "Humanities",
    "legal_professional": "Humanities", "logic": "Humanities",
    "middle_school_history": "Humanities", "modern_chinese_history": "Humanities",
    "professional_tour_guide": "Humanities",
    # ---- Other ----
    "accountant": "Other", "basic_medicine": "Other", "civil_servant": "Other",
    "clinical_medicine": "Other", "environmental_impact_assessment_engineer": "Other",
    "fire_engineer": "Other", "physician": "Other", "plant_protection": "Other",
    "sports_science": "Other", "tax_accountant": "Other",
    "urban_and_rural_planner": "Other",
}

CATEGORIES = ["STEM", "Social Sciences", "Humanities", "Other"]

SYSTEM = "以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。"


def _fmt_question(ex: dict, with_answer: bool) -> str:
    """把一条 CEval 样本渲染成「题干 + 四选项 + 答案：」。

    ★ 结尾固定是「答案：」且**后面不留空格**——被测 token 就紧接在这里。
      如果留了空格，模型的下一个 token 更可能是别的东西，四个字母的 logit
      就不再处在同一个"该出答案了"的语境里，比较也就失去意义。
    """
    lines = [ex["question"].strip()]
    for ch in CHOICES:
        lines.append(f"{ch}. {str(ex.get(ch, '')).strip()}")
    lines.append("答案：" + (str(ex["answer"]).strip() if with_answer else ""))
    return "\n".join(lines)


def _option_token_ids(tok) -> list[int]:
    """A/B/C/D 各自的首个 token id，并校验四者互不相同。

    ★ 这个断言不是形式主义：整套方法的前提就是"四个选项落在四个可区分的 token 上"。
      某些 tokenizer（尤其是把大写字母做特殊处理的）可能把它们合并，
      那时应当**报错**，而不是继续算出一个悄悄错掉的分数。
    """
    ids = []
    for ch in CHOICES:
        enc = tok.encode(ch, add_special_tokens=False)
        if not enc:
            raise RuntimeError(f"tokenizer 把选项 {ch!r} 编成了空序列，无法评测")
        ids.append(enc[0])
    if len(set(ids)) != len(ids):
        raise RuntimeError(
            f"A/B/C/D 的首 token id 有重复：{ids}。\n"
            f"本脚本靠比较这四个 token 的 logit 判分，id 重复时方法不成立。")
    return ids


def run_ceval(model_path: str | Path, tag: str, n_shot: int = 5,
              limit: int = 0, subjects: list[str] | None = None,
              save: bool = True, verbose: bool = True) -> dict:
    """跑一遍 CEval（val 划分），返回总分 + 四大类 + 逐学科。"""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    subs = subjects or sorted(SUBJECT_CATEGORY)
    unknown = [s for s in subs if s not in SUBJECT_CATEGORY]
    if unknown:
        raise ValueError(f"未知学科：{unknown}")

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    opt_ids = _option_token_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.eval()

    per_subject: dict[str, dict] = {}
    # 按题累计（micro），见 SUBJECT_CATEGORY 上方注释
    cat_hit = {c: 0 for c in CATEGORIES}
    cat_tot = {c: 0 for c in CATEGORIES}
    all_hit = all_tot = 0
    t0 = time.time()

    for si, sub in enumerate(subs, 1):
        ds = load_dataset("ceval/ceval-exam", name=sub)
        dev, val = ds["dev"], ds["val"]
        shots = [_fmt_question(dev[i], with_answer=True)
                 for i in range(min(n_shot, len(dev)))]
        head = SYSTEM.format(subject=sub.replace("_", " "))
        prefix = head + "\n\n" + "\n\n".join(shots) + ("\n\n" if shots else "")

        n = len(val) if not limit else min(limit, len(val))
        hit = 0
        # ★ 逐题对错也记下来（不只是聚合的 correct 计数）。
        #   理由：比较两个模型时，**配对检验**（McNemar）比"两个独立比例的 z 检验"
        #   强得多——同一批题目上，只有"一个对另一个错"的那些题携带信息，
        #   两个都对/都错的题应当被抵消掉。而配对检验需要逐题记录。
        #   第一版只存了 correct 总数，导致 Day42 分析里只能退回到保守的
        #   非配对近似（见 ceval_significance.py 的说明）。
        #   Per-question hits enable a paired McNemar test; aggregate counts
        #   only permit the much weaker unpaired approximation.
        hits: list[int] = []
        for i in range(n):
            ex = val[i]
            prompt = prefix + _fmt_question(ex, with_answer=False)
            ids = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**ids).logits[0, -1]          # 最后一个位置
            pred = CHOICES[int(torch.argmax(logits[opt_ids]))]
            ok = int(pred == str(ex["answer"]).strip().upper())
            hits.append(ok)
            hit += ok

        cat = SUBJECT_CATEGORY[sub]
        per_subject[sub] = {"category": cat, "n": n, "correct": hit,
                            "acc": round(100.0 * hit / n, 2) if n else 0.0,
                            # 紧凑存成 "0101..." 字符串：1346 题也只有 1.3KB，
                            # 存成 list[int] 的话 JSON 会膨胀到十几 KB 且极难读。
                            "hits": "".join(str(x) for x in hits)}
        cat_hit[cat] += hit; cat_tot[cat] += n
        all_hit += hit; all_tot += n
        if verbose:
            print(f"  [{si:>2}/{len(subs)}] {sub:<42} {hit:>3}/{n:<3} "
                  f"{per_subject[sub]['acc']:>6.2f}", flush=True)

    result = {
        "tag": tag,
        "model": str(model_path),
        "caliber": f"ppl-{n_shot}shot",   # ★ 口径必须跟着分数走，见文件头
        "split": "val",
        "n_subjects": len(subs),
        "n_questions": all_tot,
        "limit_per_subject": limit or None,
        "Average": round(100.0 * all_hit / all_tot, 2) if all_tot else 0.0,
        **{c: (round(100.0 * cat_hit[c] / cat_tot[c], 2) if cat_tot[c] else None)
           for c in CATEGORIES},
        "seconds": round(time.time() - t0, 1),
        "per_subject": per_subject,
    }

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"{tag}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="自给自足的 CEval 评测（ppl 口径，5-shot）")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-shot", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0,
                    help="每科最多评几题（0=全部）。冒烟用。")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    r = run_ceval(args.model, args.tag, n_shot=args.n_shot,
                  limit=args.limit, subjects=args.subjects)
    print("=" * 70)
    print(f"  {r['tag']}   口径 {r['caliber']}   {r['n_questions']} 题   {r['seconds']}s")
    print(f"  Average          {r['Average']}")
    for c in CATEGORIES:
        print(f"  {c:<16} {r[c]}")
    print(f"  → {OUT_DIR / (r['tag'] + '.json')}")


if __name__ == "__main__":
    main()
