"""
make_distill_fig.py — Week8 Day42
画 B（纯 SFT）与 C（KD 蒸馏）两组的训练曲线对比。
Plots the B (plain SFT) vs C (KD) training curves.

★ 为什么必须画 CE 分量，而不是 Trainer 记的那个 `loss`
    两组的 `loss` 口径根本不同：
      B: loss = CE
      C: loss = α·T²·KL + (1−α)·CE      （α=0.5, T=2）
    直接把两条 `loss` 画在一张图上，看到的差异**大部分来自损失函数定义不同**，
    而不是模型学得好不好。唯一可比的量是 **CE 分量**（以及它的 exp，
    即训练集困惑度）——两组的 CE 都是"在同一批 assistant token 上、
    对同一套硬标签"算出来的，定义完全一致。
    distill_kd.py 的 `log()` 因此把 ce_loss / kd_loss 分开记了下来。
    Only the CE component is comparable across arms; the raw `loss` is not,
    because the two arms optimise different objectives.

★ 另一个必须说明的口径问题：日志里的 `loss` 是**累积窗口求和**，不是每 token 均值
    实测 `loss / ce_loss` 恒等于 8（= gradient_accumulation_steps，见 B 组 96 条
    日志的比值区间 7.40~8.00，末尾不足一个完整窗口时略小）。
    也就是说 `loss` 这一列不能当成"平均每 token 损失"来读，
    要看模型学得怎么样只能看 ce_loss / ppl_train。
    The logged `loss` is summed over the accumulation window (exactly 8x ce_loss).

用法 / Usage（仓库根目录）:
    .venv/Scripts/python.exe Week8/scripts/make_distill_fig.py
产物 / Output:
    Week8/reports/figs/fig_distill_curves.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "Week8" / "deliverables" / "logs" / "distill"
OUT = ROOT / "Week8" / "reports" / "figs" / "fig_distill_curves.png"

RUNS = [("sft_baseline_ep2", "B · 纯 SFT（α=0）", "#3b7dd8"),
        ("kd_a0.5_T2.0_ep2", "C · KD 蒸馏（α=0.5, T=2）", "#d9534f")]


def load(name: str) -> list[dict]:
    p = LOGS / name / "trainer_log.jsonl"
    if not p.exists():
        sys.exit(f"[FAIL] 缺日志：{p}\n       先跑完 distill_kd.py 的对应组。")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    # ★ 最后一行是 Trainer 的收尾摘要（train_runtime / total_flos / ...），
    #   没有 ce_loss 也没有 epoch 之外的曲线字段。不过滤掉会直接 KeyError。
    return [r for r in rows if "ce_loss" in r]


def main() -> None:
    data = {n: load(n) for n, _, _ in RUNS}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=160)

    # ── (a) CE 分量：唯一跨组可比的量 ──────────────────────────────
    ax = axes[0]
    for name, label, c in RUNS:
        rs = data[name]
        ax.plot([r["epoch"] for r in rs], [r["ce_loss"] for r in rs],
                color=c, lw=1.7, label=label)
    ax.set_title("(a) CE 分量（硬标签交叉熵）\n★ 两组唯一定义一致、可直接比较的量",
                 fontsize=11)
    ax.set_xlabel("epoch"); ax.set_ylabel("CE loss")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # ── (b) 训练集困惑度 exp(CE) ─────────────────────────────────
    ax = axes[1]
    for name, label, c in RUNS:
        rs = data[name]
        ax.plot([r["epoch"] for r in rs], [r["ppl_train"] for r in rs],
                color=c, lw=1.7, label=label)
    ax.set_title("(b) 训练集困惑度 exp(CE)\n同一口径，越低说明对硬标签拟合越好",
                 fontsize=11)
    ax.set_xlabel("epoch"); ax.set_ylabel("PPL")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    # ── (c) KD 项：只有 C 组有 ───────────────────────────────────
    ax = axes[2]
    rs = data["kd_a0.5_T2.0_ep2"]
    ax.plot([r["epoch"] for r in rs], [r["kd_loss"] for r in rs],
            color="#d9534f", lw=1.7, label="C · T²·KL(教师‖学生)")
    ax.axhline(0, color="#999", lw=0.8, ls=":")
    ax.set_title("(c) KD 项 T²·KL(p_teacher ‖ p_student)\n"
                 "它在下降 = 学生的输出分布确实在向教师靠拢", fontsize=11)
    ax.set_xlabel("epoch"); ax.set_ylabel("T²·KL")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

    fig.suptitle("图　B（纯 SFT）与 C（KD 蒸馏）训练曲线对比　"
                 "—　Qwen2.5-0.5B 学生 / Qwen2.5-3B-DPO 教师 / 4649 样本 × 2 epoch",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"[ok] {OUT}")

    # 顺带把关键数字打出来，写报告时直接抄，不用自己去翻 jsonl
    print("\n=== 关键数字 ===")
    for name, label, _ in RUNS:
        rs = data[name]
        print(f"  {label}")
        print(f"    CE  首 {rs[0]['ce_loss']:.4f} → 末 {rs[-1]['ce_loss']:.4f}"
              f"   PPL 首 {rs[0]['ppl_train']:.3f} → 末 {rs[-1]['ppl_train']:.3f}")
        if rs[-1].get("kd_loss"):
            print(f"    KD  首 {rs[0]['kd_loss']:.4f} → 末 {rs[-1]['kd_loss']:.4f}")
        gn = [r["grad_norm"] for r in rs]
        print(f"    grad_norm {min(gn):.1f} ~ {max(gn):.1f}"
              f"（max_grad_norm=1.0，裁剪全程饱和）")


if __name__ == "__main__":
    main()
