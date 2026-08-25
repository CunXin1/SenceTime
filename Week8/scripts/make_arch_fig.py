"""
make_arch_fig.py — Week8 Day44 / 任务书 44.1「Pipeline 架构图」
画第 7 章的流水线架构图。
Draws the Chapter-7 pipeline architecture diagram.

★ 为什么不直接用 ASCII 图
    ch7 正文里已经有一张 ASCII 架构图，它在终端和 Markdown 里都好用。
    但报告要转 .docx —— 等宽字符画在 Word 里会因为字体回退而错位，
    而且它无法表达"跨 Windows / WSL 边界"这个本项目最关键的结构特征
    （ASCII 只能画方框和线，画不出两个底色不同的区域）。
    所以这里用 matplotlib 画一张真图：**底色区分操作系统**，
    段与段之间的实线表示文件依赖，虚线表示"可跳过"。
    A real figure can encode the OS boundary with background bands;
    ASCII cannot.

★ 中文字体
    Windows 上 matplotlib 默认字体没有中文，不设置的话所有汉字都是方框。
    与 make_report_figs.py 保持一致：微软雅黑 → 黑体 → DejaVu 回退。

用法 / Usage（仓库根目录）:
    .venv/Scripts/python.exe Week8/scripts/make_arch_fig.py
产物 / Output:
    Week8/reports/figs/fig7_1_pipeline_arch.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Week8" / "reports" / "figs" / "fig7_1_pipeline_arch.png"

C_WIN = "#eef4fb"      # Windows 侧底色
C_WSL = "#fdf1e6"      # WSL 侧底色
C_STEP = "#ffffff"
C_EDGE = "#33445c"
C_ART = "#f3f6f3"
C_CTRL = "#e8eef7"


def box(ax, x, y, w, h, text, fc=C_STEP, ec=C_EDGE, fs=9.5, weight="normal", lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color="#1a2433", zorder=4, linespacing=1.5)


def arrow(ax, p0, p1, style="-", color=C_EDGE, lw=1.6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=style,
                                 shrinkA=2, shrinkB=2, zorder=5))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 7.2), dpi=170)
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 7.2); ax.axis("off")

    # ── 两个操作系统的底色带 ──────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.15, 1.55), 9.15, 4.35,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=0, facecolor=C_WIN, zorder=0))
    ax.add_patch(FancyBboxPatch((9.45, 1.55), 2.8, 4.35,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                linewidth=0, facecolor=C_WSL, zorder=0))
    ax.text(0.32, 5.66, "Windows 11 · .venv (训练栈)", fontsize=9.5,
            color="#4a5f7a", fontweight="bold", zorder=1)
    ax.text(9.62, 5.66, "WSL2 · ~/venvs/vllm", fontsize=9.5,
            color="#a06a35", fontweight="bold", zorder=1)

    # ── 标题 ────────────────────────────────────────────────────────
    ax.text(6.2, 6.82, "图 7-1　Week8 全链路 Pipeline 架构",
            ha="center", fontsize=14.5, fontweight="bold", color="#16202e")
    ax.text(6.2, 6.44, "四段单向管道 · 段间只通过文件耦合 · 跨 Windows / WSL 边界",
            ha="center", fontsize=10, color="#5b6b80")

    # ── 配置层 ──────────────────────────────────────────────────────
    box(ax, 3.35, 5.95, 5.7, 0.42,
        "Week8/configs/pipeline.env　—　纯 POSIX sh，唯一变量契约",
        fc=C_CTRL, fs=9.5, weight="bold")

    # ── 四段 ────────────────────────────────────────────────────────
    steps = [
        (0.45, "step1  数据准备\nstep1_data_prep.py"),
        (2.75, "step2  训练\nstep2_train.sh"),
        (5.05, "step3  评测\nstep3_eval.py"),
        (9.70, "step4  部署\nstep4_deploy.sh"),
    ]
    for x, t in steps:
        box(ax, x, 4.15, 2.05, 0.95, t, fs=9.8, weight="bold")

    # step4 的执行体在 WSL 里
    box(ax, 9.70, 3.05, 2.05, 0.68,
        "vLLM serve\n(AWQ / GPTQ / FP16 / VL)", fc="#fff8f0", fs=8.8)
    box(ax, 6.35, 3.05, 1.9, 0.68, "Gradio UI\n:7860", fc="#f5f9ff", fs=8.8)

    # ── 段间箭头 ────────────────────────────────────────────────────
    arrow(ax, (2.50, 4.63), (2.75, 4.63))
    arrow(ax, (4.80, 4.63), (5.05, 4.63))
    arrow(ax, (7.10, 4.63), (9.70, 4.63), style=(0, (5, 3)))
    ax.text(8.40, 4.80, "--with-deploy\n(默认关)", ha="center", fontsize=8,
            color="#8a6a45", linespacing=1.3)

    # step4 → 两个服务。
    # ★ 两条箭头的标注不能混：起 vLLM 是 `wsl.exe -e bash -lc` 投递进 WSL，
    #   起 Gradio 是本地 nohup。第一版把 wsl.exe 标在了 step4→Gradio 那条线上，
    #   正好把本项目最关键的结构特征（跨系统边界在哪一侧）标反了。
    arrow(ax, (10.40, 4.15), (10.40, 3.73), lw=1.4, color="#a06a35")
    ax.text(10.52, 3.92, "wsl.exe -e bash -lc", ha="left", va="center",
            fontsize=7.4, color="#a06a35")
    arrow(ax, (9.85, 4.28), (8.25, 3.58), lw=1.3)
    ax.text(8.92, 4.05, "nohup .venv/python", ha="center", fontsize=7.4,
            color="#5b6b80", rotation=19)
    # Gradio -> vLLM 的 OpenAI 兼容调用
    arrow(ax, (8.25, 3.39), (9.70, 3.39), lw=1.2, color="#8899ad")
    ax.text(8.97, 3.18, "OpenAI 兼容 :8000\nhttpx trust_env=False", ha="center",
            fontsize=7.4, color="#7788a0", linespacing=1.3)

    # ── 产物层 ──────────────────────────────────────────────────────
    arts = [
        (0.45, "Week8/data/\ndataset_info.json\ndata_stats.json"),
        (2.75, "saves/week8/\nmodels/*-merged\nretry_history.json"),
        (5.05, "eval_summary.csv\neval_details/\nceval/*.json"),
    ]
    for x, t in arts:
        box(ax, x, 2.10, 2.05, 0.82, t, fc=C_ART, ec="#9aa8b8", fs=8.2, lw=1.0)
        arrow(ax, (x + 1.02, 4.15), (x + 1.02, 2.92), lw=1.2, color="#9aa8b8")

    # 产物 → 下一段（文件耦合）
    for x0, x1 in ((2.50, 2.75), (4.80, 5.05)):
        ax.plot([x0, x1], [2.51, 2.51], color="#9aa8b8", lw=1.1, zorder=2)
    ax.text(4.7, 1.80, "段与段之间只通过文件耦合 —— 四段分属 Python / Bash 两种语言、"
                       "两个操作系统，任何进程内状态都传不过去",
            ha="center", fontsize=8.6, color="#6b7a8d", style="italic")

    # ── 主控 ────────────────────────────────────────────────────────
    box(ax, 3.05, 0.42, 6.3, 0.82,
        "run_pipeline.sh　主控编排\n"
        "--skip-data / --skip-train / --skip-eval / --with-deploy · "
        "--only · --from · --quick · --dry-run · --list",
        fc=C_CTRL, fs=9.0, weight="bold")
    for x, _ in steps:
        # ★ step4 那一列在 y=3.05~3.73 有 vLLM 方框，控制线不能走中轴，
        #   否则会从方框正中穿过去。绕到该列右侧。
        cx = x + 1.85 if x > 9 else x + 1.02
        arrow(ax, (cx, 1.24), (cx, 4.15) if x > 9 else (cx, 2.10),
              lw=1.0, color="#7f93ad", style=(0, (3, 3)))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"[ok] {OUT}")


if __name__ == "__main__":
    main()
