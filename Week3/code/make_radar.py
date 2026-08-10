"""
make_radar.py — Week3 Day14
读取回收的盲测打分 CSV，计算 5 维均分与加权总分，绘制雷达图并标记最优模型。
Read the collected blind-test scores, compute per-dimension means and the
weighted total, draw radar charts, and mark the best model.

输入 / Input :
    Week3/deliverables/盲测打分记录.csv   打分人填好的 CSV（多人打分同题取平均）
    Week3/deliverables/盲测映射.json      匿名 -> 真实模型映射（此时可以公开）
输出 / Output:
    Week3/deliverables/radar_overview.png    全模型叠加雷达图
    Week3/deliverables/radar_<run_id>.png    每模型单图
    Week3/deliverables/盲测得分汇总.md        维度均分 + 加权总分表 + 最优标记

用法 / Usage（仓库根目录 / from repo root）:
    .venv/Scripts/python.exe Week3/code/make_radar.py
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无窗口环境直接出图 / headless rendering
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DELIV = ROOT / "Week3" / "deliverables"
CSV_PATH = DELIV / "盲测打分记录.csv"
MAP_PATH = DELIV / "盲测映射.json"

# 维度与权重（与 人工评分卡.md 一致）/ dimensions & weights (match the rubric)
DIMS = ["准确性", "完整性", "逻辑性", "安全性", "格式"]
WEIGHTS = {"准确性": 0.30, "完整性": 0.25, "逻辑性": 0.20,
           "安全性": 0.15, "格式": 0.10}

# Windows 中文字体：微软雅黑优先，黑体兜底 / CJK fonts available on Windows
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_scores() -> dict:
    """读 CSV -> {匿名模型: {维度: [分数...]}}；空格子跳过（未打完可先看部分结果）。
    Load CSV -> {anon_model: {dim: [scores...]}}; blank cells are skipped so a
    partially-filled sheet still produces a preview."""
    scores = defaultdict(lambda: defaultdict(list))
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for d in DIMS:
                cell = (row.get(d) or "").strip()
                if cell:
                    v = float(cell)
                    if not 1 <= v <= 5:
                        raise ValueError(
                            f"分数超界 {v}（应为 1~5）：{row['匿名模型']} {row['题目ID']} {d}")
                    scores[row["匿名模型"]][d].append(v)
    if not scores:
        raise SystemExit("CSV 里还没有任何分数 / the CSV has no scores yet")
    return scores


def aggregate(scores: dict, mapping: dict) -> list:
    """算每模型维度均分与加权总分，按总分降序。
    Per-model dimension means + weighted total, sorted descending."""
    inv = {v: k for k, v in mapping.items()}  # 匿名 -> 真实 / anon -> real
    rows = []
    for anon, dims in scores.items():
        means = {d: (sum(dims[d]) / len(dims[d]) if dims[d] else 0.0)
                 for d in DIMS}
        total = sum(means[d] * WEIGHTS[d] for d in DIMS)
        rows.append({"anon": anon, "real": inv.get(anon, anon),
                     "means": means, "total": round(total, 3),
                     "n": max((len(v) for v in dims.values()), default=0)})
    rows.sort(key=lambda r: r["total"], reverse=True)
    return rows


def radar_axes(ax, means: dict, label: str, color=None, floor: float = 0.0):
    """把一个模型的 5 维分画到极坐标轴上。Plot one model onto a polar axis.

    floor：径向轴下限。分数集中在高分段时，从 0/1 起画会把所有多边形挤在
    外圈看不出差异；把下限抬到最低分附近可放大区分度（分数本身不变，
    图上注明起点）。 Radial-axis lower bound: raising it near the minimum
    score spreads out clustered high scores (values unchanged, noted on chart)."""
    angles = [2 * math.pi * i / len(DIMS) for i in range(len(DIMS))]
    values = [means[d] for d in DIMS]
    ax.plot(angles + angles[:1], values + values[:1],
            label=label, linewidth=2.2, color=color)
    ax.fill(angles + angles[:1], values + values[:1], alpha=0.10, color=color)
    ax.set_xticks(angles)
    ax.set_xticklabels(DIMS)
    ax.set_ylim(floor, 5)
    ticks, t = [], floor
    while t <= 5.001:
        ticks.append(round(t, 1))
        t += 0.5
    ax.set_yticks(ticks)
    ax.tick_params(axis="y", labelsize=8)


def main() -> None:
    mapping = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    rows = aggregate(load_scores(), mapping)
    best = rows[0]

    # 径向轴下限 = 全局最低维度分再往下留 0.25 的边距，向下取到 0.5 档；
    # 所有图共用同一下限，跨图可比。
    # Shared radial floor: 0.25 below the global minimum, snapped to 0.5 steps.
    gmin = min(r["means"][d] for r in rows for d in DIMS)
    floor = max(0.0, math.floor((gmin - 0.25) * 2) / 2)

    # ---- 总览雷达图（全模型叠加）/ overview radar (all models) ----
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, polar=True)
    for r in rows:
        lbl = f"{r['real']} ({r['total']:.2f})"
        if r is best:
            lbl = "★ " + lbl  # 最优标记 / mark the winner
        radar_axes(ax, r["means"], lbl, floor=floor)
    ax.set_title(f"Week3 盲测 5 维得分总览（括号内为加权总分；径向轴 {floor:g}~5）",
                 pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.1), fontsize=8)
    fig.savefig(DELIV / "radar_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 每模型单图 / one radar per model ----
    for r in rows:
        fig = plt.figure(figsize=(5.5, 5))
        ax = fig.add_subplot(111, polar=True)
        radar_axes(ax, r["means"], r["real"], color="#c04851", floor=floor)
        star = "★最优 " if r is best else ""
        ax.set_title(f"{star}{r['real']}  加权总分 {r['total']:.2f}"
                     f"（径向轴 {floor:g}~5）", pad=20)
        fig.savefig(DELIV / f"radar_{r['real']}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # ---- 得分汇总表 / score summary ----
    lines = ["# Week3 盲测得分汇总", "",
             "> 由 make_radar.py 自动生成；权重：准确30/完整25/逻辑20/安全15/格式10。",
             "> ★ = 加权总分最高（人工盲测维度的最优模型）。", "",
             "| 模型 | " + " | ".join(DIMS) + " | 加权总分 | 打分题数 |",
             "|---|" + "---|" * (len(DIMS) + 2)]
    for r in rows:
        star = " ★" if r is best else ""
        cells = " | ".join(f"{r['means'][d]:.2f}" for d in DIMS)
        lines.append(f"| `{r['real']}`{star} | {cells} | "
                     f"**{r['total']:.2f}** | {r['n']} |")
    (DELIV / "盲测得分汇总.md").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")

    print(f"[OK] 最优 / best: {best['real']}  total={best['total']:.2f}")
    print(f"[OK] radar_overview.png + {len(rows)} per-model charts + 盲测得分汇总.md")


if __name__ == "__main__":
    main()
