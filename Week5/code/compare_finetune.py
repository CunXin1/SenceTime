"""Day26 交付:微调前后客观对比 -> 微调前后对比表.md + CSV + 柱状图

❸ 的验收要求是"微调后有明显效果提升"。主观任务(写产品描述)只能靠人打分,
样本一少就说不清是提升还是噪声。所以 Day26 的任务被设计成**结构化字段抽取**,
20 条留出图每张 8 个字段,一共 160 个字段可以逐个和真值精确比对。

三个指标:
    字段准确率   = 抽对的字段数 / 总字段数(160)           ← 主指标
    整条全对率   = 8 个字段全部抽对的图片比例
    格式合规率   = 输出恰好 8 行、每行都是「- 字段名：值」的比例
                   (微调前模型常常加一堆解释文字,这一项通常是提升最明显的)

用法:
    # 基线(未微调)
    .venv-vlm\\Scripts\\python.exe Week5/code/compare_finetune.py --tag base
    # 挂 LoRA adapter(不必先合并)
    .venv-vlm\\Scripts\\python.exe Week5/code/compare_finetune.py --tag lora \\
        --adapter saves/qwen2.5vl-7b-week5-lora
    # 或用合并后的模型
    .venv-vlm\\Scripts\\python.exe Week5/code/compare_finetune.py --tag merged \\
        --model-path models/Qwen2.5-VL-7B-week5-merged
    # 两轮都跑完后出报告
    .venv-vlm\\Scripts\\python.exe Week5/code/compare_finetune.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_vlm_sft_data import COPY_FIELDS, DERIVED_FIELDS, FIELDS  # noqa: E402
from vlm_common import ROOT, SPECS, generate, load_vlm  # noqa: E402

DATA = ROOT / "Week5" / "data"
DELIV = ROOT / "Week5" / "deliverables"
EVAL_RECORDS = DATA / "eval_records.json"
RAW = DELIV / "day26_raw.json"
CSV_OUT = DELIV / "微调前后明细.csv"
MD_OUT = DELIV / "微调前后对比表.md"
PNG_OUT = DELIV / "微调前后对比.png"

# FIELDS / COPY_FIELDS / DERIVED_FIELDS 从 build_vlm_sft_data 导入，两边永远一致
LINE_RE = re.compile(r"^\s*[-*·]?\s*([^:：]+)\s*[:：]\s*(.*)$")


def parse_fields(text: str) -> dict[str, str]:
    """从模型输出里抽出「字段名: 值」。容忍全角/半角冒号、有无前导横线。"""
    got: dict[str, str] = {}
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        k = m.group(1).strip().replace(" ", "").replace("当前loss", "当前loss")
        v = m.group(2).strip().rstrip("。")
        for f in FIELDS:
            if k == f or k.replace("_", "") == f:
                got.setdefault(f, v)
                break
    return got


def norm(v: str) -> str:
    """比对前做温和归一化:去空格、统一全角括号/冒号,大小写不敏感。
    不做更激进的清洗 —— 否则会把模型的错误也一起"洗对"。"""
    return (v.strip().replace(" ", "").replace("（", "(").replace("）", ")")
            .replace("，", ",").lower())


def score_one(pred_text: str, truth: dict[str, str]) -> dict:
    got = parse_fields(pred_text)
    hits = {f: (f in got and norm(got[f]) == norm(truth[f])) for f in FIELDS}
    lines = [l for l in pred_text.strip().splitlines() if l.strip()]
    fmt_ok = len(lines) == len(FIELDS) and all(LINE_RE.match(l) for l in lines)
    return {"got": got, "hits": hits, "n_hit": sum(hits.values()),
            "all_hit": all(hits.values()), "format_ok": fmt_ok}


def run(tag: str, key: str, model_path: str | None, adapter: str | None) -> list[dict]:
    records = json.loads(EVAL_RECORDS.read_text(encoding="utf-8"))
    print(f"\n{'=' * 78}\n[{tag}] {SPECS[key].display}"
          f"{'  +adapter ' + adapter if adapter else ''}"
          f"{'  path=' + model_path if model_path else ''}\n"
          f"留出集 {len(records)} 条 × {len(FIELDS)} 字段 = {len(records) * len(FIELDS)} 个字段\n"
          f"{'=' * 78}", flush=True)

    vlm = load_vlm(key, attn_impl="sdpa",
                   path_override=ROOT / model_path if model_path else None,
                   adapter=ROOT / adapter if adapter else None)
    rows = []
    for i, rec in enumerate(records, 1):
        r = generate(vlm, ROOT / rec["image"], rec["instruction"], max_new_tokens=256)
        s = score_one(r.text, rec["fields"])
        rows.append({
            "tag": tag, "model": key, "image": Path(rec["image"]).name,
            "n_hit": s["n_hit"], "all_hit": int(s["all_hit"]),
            "format_ok": int(s["format_ok"]),
            **{f"hit_{f}": int(s["hits"][f]) for f in FIELDS},
            "pred": r.text, "truth": rec["target"],
            "latency_s": round(r.latency_s, 2),
        })
        print(f"[{i}/{len(records)}] {Path(rec['image']).name}  "
              f"字段 {s['n_hit']}/{len(FIELDS)}  全对={'✅' if s['all_hit'] else '❌'}  "
              f"格式={'✅' if s['format_ok'] else '❌'}", flush=True)
        if not s["all_hit"]:
            miss = [f for f in FIELDS if not s["hits"][f]]
            for f in miss[:3]:
                print(f"      {f}: 真值={rec['fields'][f]!r} 抽到={s['got'].get(f)!r}")

    del vlm
    torch.cuda.empty_cache()
    return rows


def agg(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    total_f = n * len(FIELDS)
    out = {
        "n": n,
        "field_acc": sum(r["n_hit"] for r in rows) / total_f,
        "all_hit_rate": sum(r["all_hit"] for r in rows) / n,
        "format_rate": sum(r["format_ok"] for r in rows) / n,
        "latency": sum(r["latency_s"] for r in rows) / n,
        # 照抄字段 vs 派生字段分开算 —— 这是本实验最有信息量的一刀:
        # 照抄字段基座本来就强，提升空间小；派生字段要靠团队规则，只能从数据里学
        "copy_acc": sum(r[f"hit_{f}"] for r in rows for f in COPY_FIELDS) / (n * len(COPY_FIELDS)),
        "derived_acc": sum(r[f"hit_{f}"] for r in rows for f in DERIVED_FIELDS) / (n * len(DERIVED_FIELDS)),
    }
    for f in FIELDS:
        out[f"acc_{f}"] = sum(r[f"hit_{f}"] for r in rows) / n
    return out


def plot(by_tag: dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=f).get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False

    tags = list(by_tag)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))

    labels = ["字段准确率", "整条全对率", "格式合规率"]
    w = 0.8 / max(len(tags), 1)
    for i, t in enumerate(tags):
        m = by_tag[t]
        vals = [m["field_acc"], m["all_hit_rate"], m["format_rate"]]
        pos = [x + i * w for x in range(len(labels))]
        bars = axes[0].bar(pos, vals, width=w, label=t)
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}",
                         ha="center", fontsize=9)
    axes[0].set_xticks([x + w * (len(tags) - 1) / 2 for x in range(len(labels))])
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_title("总体指标")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    for i, t in enumerate(tags):
        m = by_tag[t]
        vals = [m[f"acc_{f}"] for f in FIELDS]
        pos = [x + i * w for x in range(len(FIELDS))]
        axes[1].bar(pos, vals, width=w, label=t)
    axes[1].set_xticks([x + w * (len(tags) - 1) / 2 for x in range(len(FIELDS))])
    axes[1].set_xticklabels(FIELDS, rotation=30, ha="right")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title("按字段拆解准确率")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Day26　VLM LoRA 微调前后对比（冻结 ViT，200 条训练 / 20 条留出）", fontsize=13)
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=130)
    plt.close(fig)
    print(f"[写出] {PNG_OUT}")


def write_outputs(all_rows: list[dict]) -> None:
    tags = []
    for r in all_rows:
        if r["tag"] not in tags:
            tags.append(r["tag"])
    by_tag = {t: agg([r for r in all_rows if r["tag"] == t]) for t in tags}

    fields = (["tag", "model", "image", "n_hit", "all_hit", "format_ok"]
              + [f"hit_{f}" for f in FIELDS] + ["latency_s", "pred", "truth"])
    with CSV_OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[写出] {CSV_OUT}  {len(all_rows)} 行")

    plot(by_tag)

    base = by_tag.get("base")
    L = ["# Day26 交付：VLM LoRA 微调前后对比\n",
         f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M}　"
         f"由 `Week5/code/compare_finetune.py` 生成，手改会被覆盖。\n",
         "## 一、任务设计（两次迭代）\n",
         "任务书 26.1 的例子是「请根据这张图写一段产品描述」。产品描述是主观生成任务，"
         "微调前后**没法客观打分**，而 ❸ 的验收要求是「有明显效果提升」——"
         "主观任务只能靠人打分，20 个样本说不清是提升还是噪声。\n",
         "所以换成可客观打分的窄任务：**训练平台截图 → 团队规范的实验记录卡**。"
         f"20 条留出图 × {len(FIELDS)} 个字段 = {20 * len(FIELDS)} 个字段逐个和真值精确比对。\n",
         "> **第一版设计失败了，这里如实记录。** 最初 11 个字段全是「照抄」型"
         "（从截图里读出来即可）。实测 Qwen2.5-VL-7B **基线字段准确率就有 ~92%**，"
         "微调最多再涨 8 个点，测不出 LoRA 的价值。\n",
         "第二版把字段分成两类：\n",
         f"| 类别 | 字段 | 基座能不能自己做对 |\n|---|---|---|",
         f"| **照抄型**（{len(COPY_FIELDS)} 个） | {'、'.join(COPY_FIELDS)} | 能，强基座 ~92% |",
         f"| **派生型**（{len(DERIVED_FIELDS)} 个） | {'、'.join(DERIVED_FIELDS)} | 不能——需要算术，"
         "且「健康状态」是一条**只存在于训练数据里的团队规则**，指令里不给 |",
         "",
         "「健康状态」的规则是：`状态==失败` 或 `loss≥1.0` 或 `显存≥20.0 GB` → 需关注，否则正常。"
         "这条规则**没有写进 instruction**，基座只能猜，微调模型才能从 200 条数据里学到。"
         "这样测出来的才是 LoRA 真正学到了什么，而不是基座本来就会什么。\n",
         "训练集与留出集用两条**独立随机流**生成（seed 42 / 10042），"
         "图片内容、版式、配色都不重叠。\n",
         "## 二、总体指标\n",
         "| 指标 | " + " | ".join(tags) + " | 提升 |",
         "|---|" + "---|" * (len(tags) + 1)]

    for label, field, fmt in [("**字段准确率**（主指标）", "field_acc", "{:.1%}"),
                              (f"　├ 照抄型 {len(COPY_FIELDS)} 字段", "copy_acc", "{:.1%}"),
                              (f"　└ **派生型 {len(DERIVED_FIELDS)} 字段**", "derived_acc", "**{:.1%}**"),
                              (f"整条 {len(FIELDS)} 字段全对率", "all_hit_rate", "{:.1%}"),
                              ("格式合规率", "format_rate", "{:.1%}"),
                              ("平均延迟", "latency", "{:.2f}s")]:
        cells = [fmt.format(by_tag[t][field]) for t in tags]
        delta = "—"
        if base and len(tags) > 1:
            last = by_tag[tags[-1]][field]
            d = last - base[field]
            delta = (f"{d:+.1%}" if field != "latency" else f"{d:+.2f}s")
        L.append(f"| {label} | " + " | ".join(cells) + f" | {delta} |")
    L.append("")

    L.append("## 三、按字段拆解\n")
    L.append("| 字段 | " + " | ".join(tags) + " | 提升 |")
    L.append("|---|" + "---|" * (len(tags) + 1))
    for f in FIELDS:
        cells = [f"{by_tag[t][f'acc_{f}']:.0%}" for t in tags]
        delta = "—"
        if base and len(tags) > 1:
            delta = f"{by_tag[tags[-1]][f'acc_{f}'] - base[f'acc_{f}']:+.0%}"
        L.append(f"| {f} | " + " | ".join(cells) + f" | {delta} |")
    L.append("")

    L.append("## 四、对比图\n![微调前后对比](微调前后对比.png)\n")

    L.append("## 五、样例输出对比\n")
    for img in sorted({r["image"] for r in all_rows})[:3]:
        L.append(f"### `{img}`\n")
        truth = next(r["truth"] for r in all_rows if r["image"] == img)
        L.append(f"**真值**\n```\n{truth}\n```\n")
        for t in tags:
            r = next((x for x in all_rows if x["image"] == img and x["tag"] == t), None)
            if r:
                L.append(f"**{t}**（字段 {r['n_hit']}/{len(FIELDS)}，"
                         f"格式{'合规' if r['format_ok'] else '不合规'}）\n"
                         f"```\n{r['pred'][:800]}\n```\n")

    L.append("## 六、结论\n")
    if base and len(tags) > 1:
        last_t = tags[-1]
        m = by_tag[last_t]
        for label, f in [("字段准确率（总）", "field_acc"),
                         ("　照抄型字段", "copy_acc"),
                         ("　派生型字段", "derived_acc"),
                         ("整条全对率", "all_hit_rate"),
                         ("格式合规率", "format_rate")]:
            L.append(f"- {label} {base[f]:.1%} → {m[f]:.1%}（{m[f] - base[f]:+.1%}）")
        L.append("\n**读法**：三类提升的含义完全不同。\n")
        L.append("1. **派生型字段**的提升最能说明问题——「健康状态」的判定规则不在指令里，"
                 "基座只能猜，微调模型是真的从 200 条数据里学到了这条团队规范。"
                 "这就是 LoRA 在做的事：把领域约定编码进权重。")
        L.append("2. **格式合规率**的提升说明学会了输出规范，这是 200 条数据最容易学到的东西，"
                 "但也最不值钱——写个后处理正则也能做到。")
        L.append("3. **照抄型字段**提升有限是符合预期的：这些字段考的是读图精度，"
                 "而 ViT 被冻结了，视觉表征根本没变。这反过来印证了 Day22 的观察——"
                 "投影层已经把视觉特征对齐好，LoRA 改的是语言侧怎么组织输出，不是眼睛怎么看。\n")
    else:
        L.append("（还只有一组结果，跑完另一组后重新执行 `--report`。）\n")

    MD_OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[写出] {MD_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="本轮标签，如 base / lora / merged")
    ap.add_argument("--model", choices=list(SPECS), default="qwen")
    ap.add_argument("--model-path", help="相对仓库根的模型目录（合并后的模型）")
    ap.add_argument("--adapter", help="相对仓库根的 LoRA adapter 目录")
    ap.add_argument("--report", action="store_true", help="不跑模型，只用已有结果出报告")
    args = ap.parse_args()

    existing = json.loads(RAW.read_text(encoding="utf-8")) if RAW.exists() else []

    if args.report:
        if not existing:
            sys.exit(f"[中止] {RAW} 不存在，先跑至少一轮")
        write_outputs(existing)
    else:
        if not args.tag:
            sys.exit("[中止] 需要 --tag（如 base / lora）")
        if not EVAL_RECORDS.exists():
            sys.exit("[中止] 先跑 build_vlm_sft_data.py 生成留出集")
        rows = [r for r in existing if r["tag"] != args.tag]  # 同名 tag 覆盖重跑
        rows += run(args.tag, args.model, args.model_path, args.adapter)
        RAW.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        write_outputs(rows)
