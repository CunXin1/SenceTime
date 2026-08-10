"""
collect_dpo_results.py — Week4 Day19-20
汇总 3 组 DPO 实验的 rewards/loss/耗时/显存，生成对比表并归档原始日志。
Aggregate the 3 DPO runs' rewards/loss/time/VRAM into a comparison table and
archive raw logs. 复刻 Week3 collect_results.py，指标换成 DPO 的 rewards/*。

DPO 指标（键名见 LLaMA-Factory/src/llamafactory/train/dpo/trainer.py）:
    训练日志（trainer_state.json 的 log_history）:
        rewards/chosen  应上升 / should rise
        rewards/rejected 应下降 / should fall
        rewards/margins  = chosen - rejected，应扩大 / should widen
        rewards/accuracies chosen>rejected 的比例，应 → 0.9+ / should approach 0.9+
    取每条指标的 first（首个 log）/ last（末个 log）/ best（最优）三个值判断趋势。

输出 / Output:
    Week4/deliverables/DPO实验结果汇总.md   自动生成对比表（每次运行整体重写）
    Week4/deliverables/logs/<run_id>/        原始日志归档（--copy-logs 时）

用法 / Usage（仓库根目录 / from repo root）:
    .venv/Scripts/python.exe Week4/code/collect_dpo_results.py --copy-logs
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "Week4" / "configs" / "exp" / "experiments.json"
OUT_MD = ROOT / "Week4" / "deliverables" / "DPO实验结果汇总.md"
LOG_DIR = ROOT / "Week4" / "deliverables" / "logs"

LOG_FILES = ["trainer_log.jsonl", "train_results.json", "trainer_state.json",
             "training_loss.png", "training_rewards_accuracies.png",
             "run_meta.json", "console.log"]

# 训练轨迹要提取的 rewards 指标 / reward metrics to trace
REWARD_KEYS = ["rewards/chosen", "rewards/rejected", "rewards/margins",
               "rewards/accuracies"]


def load_json(p: Path):
    """容错读 JSON：缺失返回 None。Fault-tolerant JSON load."""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def trace(trainer_state: dict, key: str) -> tuple:
    """从 log_history 提取某指标的 (first, last, best)。
    Extract (first, last, best) of one metric from the training log history.
    best：chosen/margins/accuracies 取 max，rejected 取 min（越低越好）。"""
    if not trainer_state:
        return None, None, None
    vals = [h[key] for h in trainer_state.get("log_history", []) if key in h]
    if not vals:
        return None, None, None
    best = min(vals) if key == "rewards/rejected" else max(vals)
    return vals[0], vals[-1], best


def eval_final(trainer_state: dict, key: str):
    """取末次评估的 eval_ 指标（如 eval_rewards/accuracies、eval_loss）。
    Final eval metric (e.g. eval_rewards/accuracies, eval_loss)."""
    if not trainer_state:
        return None
    evs = [h[key] for h in trainer_state.get("log_history", []) if key in h]
    return evs[-1] if evs else None


def collect_run(exp: dict) -> dict:
    """收集单个 DPO 实验的全部指标。Collect every metric for one DPO run."""
    out = ROOT / exp["output_dir"]
    tr = load_json(out / "train_results.json")
    ts = load_json(out / "trainer_state.json")
    meta = load_json(out / "run_meta.json")
    row = {**exp, "done": tr is not None,
           "train_loss": tr.get("train_loss") if tr else None,
           "wall": meta.get("wall_pretty") if meta else None,
           "eta10": meta.get("eta_at_10pct") if meta else None,
           "peak_vram_gb": round(meta["peak_vram_mib"] / 1024, 1)
           if meta and meta.get("peak_vram_mib") else None,
           "eval_acc": eval_final(ts, "eval_rewards/accuracies"),
           "eval_margin": eval_final(ts, "eval_rewards/margins")}
    for key in REWARD_KEYS:
        f, l, b = trace(ts, key)
        short = key.split("/")[1]           # chosen/rejected/margins/accuracies
        row[f"{short}_first"] = f
        row[f"{short}_last"] = l
        row[f"{short}_best"] = b
    return row


def fmt(v, nd=4):
    """None -> '—'。Format a cell."""
    if v is None:
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def trend(first, last, want_up: bool) -> str:
    """趋势符号：符合期望 ✅、相反 ⚠️、无数据 —。Trend glyph vs expectation."""
    if first is None or last is None:
        return "—"
    rose = last > first
    ok = rose if want_up else not rose
    arrow = "↑" if rose else "↓"
    return f"{arrow} {'✅' if ok else '⚠️'}"


def group_rows(rows: list, group: str) -> list:
    """基线 groups='A+B' 同时属于 A 和 B。Baseline belongs to every group."""
    return [r for r in rows if group in r["groups"]]


def render() -> list:
    exps = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = [collect_run(e) for e in exps]
    done = sum(r["done"] for r in rows)

    lines = [
        f"# Week4 DPO 实验结果汇总（{len(rows)} 组对比）",
        "",
        "> 本文件由 `Week4/code/collect_dpo_results.py` 自动生成，手改会被覆盖；",
        "> 分析结论见《第4周_DPO偏好对齐报告.md》。",
        f"> 完成进度 / progress: **{done}/{len(rows)}**。",
        "",
        "## 一、Rewards 趋势总表（趋势判定：符合期望 ✅ / 相反 ⚠️）",
        "",
        "期望：`rewards/chosen` ↑、`rewards/rejected` ↓、`margins` ↑、`accuracies` ↑。",
        "",
        "| run_id | β | lr | chosen 首→末 | rejected 首→末 | margins 首→末 | "
        "acc 首→末 | 趋势 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        tr_c = trend(r["chosen_first"], r["chosen_last"], want_up=True)
        tr_r = trend(r["rejected_first"], r["rejected_last"], want_up=False)
        tr_m = trend(r["margins_first"], r["margins_last"], want_up=True)
        tr_a = trend(r["accuracies_first"], r["accuracies_last"], want_up=True)
        allok = all("✅" in t for t in (tr_c, tr_r, tr_m, tr_a))
        summary = "✅ 正确" if allok else ("⏳" if not r["done"] else "⚠️ 部分异常")
        lines.append(
            f"| `{r['run_id']}` | {r['pref_beta']} | {r['learning_rate']} | "
            f"{fmt(r['chosen_first'])}→{fmt(r['chosen_last'])} {tr_c} | "
            f"{fmt(r['rejected_first'])}→{fmt(r['rejected_last'])} {tr_r} | "
            f"{fmt(r['margins_first'])}→{fmt(r['margins_last'])} {tr_m} | "
            f"{fmt(r['accuracies_first'],3)}→{fmt(r['accuracies_last'],3)} {tr_a} | "
            f"{summary} |")

    lines += ["", "## 二、末次评估 + 训练成本", "",
              "| run_id | eval acc | eval margin | train loss | 耗时 | "
              "峰值显存 | 10%时预估 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        vram = f"{r['peak_vram_gb']} GB" if r["peak_vram_gb"] else "—"
        lines.append(
            f"| `{r['run_id']}` | {fmt(r['eval_acc'],3)} | {fmt(r['eval_margin'])} | "
            f"{fmt(r['train_loss'])} | {fmt(r['wall'])} | {vram} | {fmt(r['eta10'])} |")

    # 组内对比小结（A=β、B=lr）
    lines += ["", "## 三、控制变量对比", ""]
    for g, gtitle, var in [("A", "pref_beta（KL 约束强度）", "pref_beta"),
                           ("B", "learning_rate（学习率）", "learning_rate")]:
        grows = group_rows(rows, g)
        lines += [f"### 组{g}：{gtitle}", "",
                  "| run_id | 变量值 | 末 margins | 末 acc | 趋势正确 |",
                  "|---|---|---|---|---|"]
        for r in grows:
            allok = (all("✅" in trend(r[f"{k}_first"], r[f"{k}_last"],
                                       want_up=(k != "rejected"))
                         for k in ["chosen", "rejected", "margins", "accuracies"])
                     if r["done"] else False)
            lines.append(
                f"| `{r['run_id']}` | {r[var]} | {fmt(r['margins_last'])} | "
                f"{fmt(r['accuracies_last'],3)} | {'✅' if allok else '⏳/⚠️'} |")
        lines.append("")
    return rows, lines


def copy_logs(rows: list) -> int:
    n = 0
    for r in rows:
        if not r["done"]:
            continue
        src = ROOT / r["output_dir"]
        dst = LOG_DIR / r["run_id"]
        dst.mkdir(parents=True, exist_ok=True)
        for name in LOG_FILES:
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
                n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--copy-logs", action="store_true",
                    help="同时归档原始日志 / also archive raw logs")
    args = ap.parse_args()

    rows, lines = render()
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    done = sum(r["done"] for r in rows)
    print(f"[OK] {OUT_MD.relative_to(ROOT)}  ({done}/{len(rows)} runs done)")

    if args.copy_logs:
        n = copy_logs(rows)
        print(f"[OK] {n} log files -> {LOG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
