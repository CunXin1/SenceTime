"""
plot_rewards.py — Week4 Day21
读各组 trainer_state.json 的 log_history，绘制 DPO rewards 曲线。
Read each run's trainer_state.json log_history and plot the DPO rewards curves.

复用 Week3 make_radar.py 的 headless matplotlib + Windows 中文字体设置。
每组一张 rewards_<run_id>.png（上：chosen/rejected + margins；下：accuracies），
外加 rewards_overview.png（各组 margins 叠加对比）。对缺失键容错。

输出 / Output:
    Week4/deliverables/rewards_<run_id>.png   每组曲线
    Week4/deliverables/rewards_overview.png    各组 margins 对比

用法 / Usage（仓库根目录 / from repo root）:
    .venv/Scripts/python.exe Week4/code/plot_rewards.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # 无窗口渲染 / headless
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "Week4" / "configs" / "exp" / "experiments.json"
DELIV = ROOT / "Week4" / "deliverables"


def series(log_history: list, key: str) -> tuple:
    """从 log_history 提取 (steps, values)（只取含该 key 的训练日志条目）。"""
    xs, ys = [], []
    for h in log_history:
        if key in h and "step" in h:
            xs.append(h["step"])
            ys.append(h[key])
    return xs, ys


def load_state(run_id: str) -> dict | None:
    p = ROOT / "saves" / "week4" / "qwen" / run_id / "trainer_state.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def plot_one(run_id: str, beta: str, lr: str) -> bool:
    ts = load_state(run_id)
    if not ts:
        print(f"[skip] {run_id}: trainer_state.json 不存在（训练未完成？）")
        return False
    hist = ts.get("log_history", [])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # 上图：rewards/chosen、rewards/rejected、margins
    for key, color, label in [
        ("rewards/chosen", "#93c47d", "chosen（应↑）"),
        ("rewards/rejected", "#e06666", "rejected（应↓）"),
        ("rewards/margins", "#3d85c6", "margins（应↑）")]:
        xs, ys = series(hist, key)
        if xs:
            ax1.plot(xs, ys, marker=".", color=color, label=label)
    ax1.axhline(0, color="#999", lw=0.8, ls="--")
    ax1.set_ylabel("reward")
    ax1.set_title(f"{run_id}  (β={beta}, lr={lr})  —  DPO rewards")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 下图：accuracies
    xs, ys = series(hist, "rewards/accuracies")
    if xs:
        ax2.plot(xs, ys, marker=".", color="#f6b26b", label="accuracies（应→1）")
    ax2.axhline(0.9, color="#e06666", lw=0.8, ls="--", label="0.9 参考线")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("training step")
    ax2.set_ylabel("accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = DELIV / f"rewards_{run_id}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[OK] {out.relative_to(ROOT)}")
    return True


def plot_overview(exps: list) -> None:
    """各组 margins 叠加对比。Overlay margins of all runs."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#3d85c6", "#6aa84f", "#a64d79"]
    any_data = False
    for exp, color in zip(exps, colors):
        ts = load_state(exp["run_id"])
        if not ts:
            continue
        xs, ys = series(ts.get("log_history", []), "rewards/margins")
        if xs:
            any_data = True
            ax.plot(xs, ys, marker=".", color=color,
                    label=f"{exp['run_id']} (β={exp['pref_beta']}, lr={exp['learning_rate']})")
    if not any_data:
        plt.close(fig)
        print("[skip] overview: 暂无可用数据")
        return
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_xlabel("training step")
    ax.set_ylabel("rewards/margins")
    ax.set_title("三组 DPO 的 margins 对比（越高区分度越强）")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = DELIV / "rewards_overview.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[OK] {out.relative_to(ROOT)}")


def main() -> None:
    exps = json.loads(MANIFEST.read_text(encoding="utf-8"))
    DELIV.mkdir(parents=True, exist_ok=True)
    n = 0
    for exp in exps:
        if plot_one(exp["run_id"], exp["pref_beta"], exp["learning_rate"]):
            n += 1
    plot_overview(exps)
    print(f"\n[done] {n}/{len(exps)} 组已出图 -> {DELIV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
