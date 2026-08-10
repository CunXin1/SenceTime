"""跑模型之前的数据自检:所有引用的图片都存在、真值无占位符、探针正负平衡、
训练集格式符合 LLaMA-Factory 要求。

在 Day23 之前跑一遍,能挡掉大部分"跑到一半才发现路径错了"的浪费。

用法:
    .venv-vlm\\Scripts\\python.exe Week5/code/validate_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Week5" / "data"
IMG = DATA / "images"

problems: list[str] = []
notes: list[str] = []


def check(cond: bool, msg: str) -> None:
    (notes if cond else problems).append(("✅ " if cond else "❌ ") + msg)


# ---------------------------------------------------------------- ground truth
gt = json.loads((IMG / "ground_truth.json").read_text(encoding="utf-8"))
check("【" not in json.dumps(gt, ensure_ascii=False),
      f"ground_truth.json 无【】占位符（{len(gt)} 张图）")
for name, e in gt.items():
    check((IMG / name).exists(), f"ground_truth 引用的 {name} 存在")
    check(bool(e.get("absent_objects")), f"{name} 有 absent_objects（Day25 幻觉判定依赖它）")
    overlap = set(e.get("present_objects", [])) & set(e.get("absent_objects", []))
    check(not overlap, f"{name} 的 present/absent 无矛盾项" + (f"，冲突：{overlap}" if overlap else ""))

# ---------------------------------------------------------------- questions
qs = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
check(all((IMG / q["image"]).exists() for q in qs), f"questions.json 全部图片就位（{len(qs)} 条）")
core = [q for q in qs if q["is_core"]]
check(len(core) == 25, f"核心问题 25 条（实际 {len(core)}）—— 交付表要求 5 图 × 5 类")
cats = {q["category"] for q in qs}
check(len(cats) == 5, f"覆盖 5 类问题：{sorted(cats)}")

# ---------------------------------------------------------------- probes
pr = json.loads((DATA / "hallucination_probes.json").read_text(encoding="utf-8"))
check(all((IMG / p["image"]).exists() for p in pr), f"探针图片全部就位（{len(pr)} 条）")
ex = [p for p in pr if p["type"] == "existence"]
ny = sum(p["gt_answer"] == "yes" for p in ex)
nn = len(ex) - ny
check(ny == nn, f"存在性探针正负平衡 {ny}:{nn}（不平衡则「一律答是/否」也能刷高分）")
sy = [p for p in pr if p["type"] == "sycophancy"]
check(all(p.get("pushback") for p in sy), f"迎合诱导探针都有 pushback（{len(sy)} 条）")
# 负样本问的东西必须真的在 absent_objects 里,否则真值就是错的
bad = [p["pid"] for p in ex if p["gt_answer"] == "no" and p.get("auto")
       and not any(o in p["question"] for o in gt[p["image"]]["absent_objects"])]
check(not bad, "自动生成的负样本都能在 absent_objects 里找到对应物体" + (f"，异常：{bad}" if bad else ""))

# ---------------------------------------------------------------- SFT 数据
sft = json.loads((DATA / "vlm_sft_200.json").read_text(encoding="utf-8"))
check(len(sft) == 200, f"训练集 200 条（实际 {len(sft)}）")
miss = [r["images"][0] for r in sft if not (ROOT / r["images"][0]).exists()]
check(not miss, f"训练图全部存在" + (f"，缺 {len(miss)} 张" if miss else ""))
check(all(r["messages"][0]["content"].startswith("<image>") for r in sft),
      "每条训练样本的 user content 都以 <image> 开头（LF 多模态模板要求）")
check(all(len(r["messages"]) == 2 for r in sft), "每条样本恰好 user/assistant 两轮")

ev = json.loads((DATA / "eval_records.json").read_text(encoding="utf-8"))
check(len(ev) == 20, f"留出集 20 条（实际 {len(ev)}）")
miss_e = [r["image"] for r in ev if not (ROOT / r["image"]).exists()]
check(not miss_e, "留出图全部存在" + (f"，缺 {len(miss_e)} 张" if miss_e else ""))
train_imgs = {r["images"][0] for r in sft}
check(not (train_imgs & {r["image"] for r in ev}), "训练集与留出集图片无重叠")

di = json.loads((DATA / "dataset_info.json").read_text(encoding="utf-8"))
ds = di.get("week5_vlm_sft", {})
check(ds.get("formatting") == "sharegpt", "dataset_info formatting=sharegpt")
check(ds.get("columns", {}).get("images") == "images",
      "dataset_info 声明了 images 列（不声明的话 LF 会忽略图片，训成纯文本 SFT）")
check((DATA / ds.get("file_name", "")).exists(), "dataset_info 指向的数据文件存在")

# ---------------------------------------------------------------- 输出
print("=" * 70)
for n in notes:
    print(n)
if problems:
    print("-" * 70)
    for p in problems:
        print(p)
print("=" * 70)
print(f"通过 {len(notes)} 项，失败 {len(problems)} 项")
sys.exit(1 if problems else 0)
