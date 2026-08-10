"""
collect_results.py — Week3 Day12-13
汇总 14 组实验的训练结果，生成对比表并归档原始日志。
Aggregate the results of all 14 runs into a comparison table and archive the
raw logs.

数据来源（每个 output_dir 下）/ Sources (inside each output_dir):
    train_results.json   final train loss / 训练器统计
    trainer_state.json   log_history -> eval loss 曲线（取 best 与 final）
    run_meta.json        run_experiments.py 记录的墙钟耗时与峰值显存
    trainer_log.jsonl / training_loss.png / console.log  原始日志

输出 / Output:
    Week3/deliverables/实验结果汇总.md      自动生成的对比表（每次运行整体重写）
    Week3/deliverables/logs/<run_id>/       原始日志归档（--copy-logs 时）

用法 / Usage（仓库根目录 / from repo root）:
    .venv/Scripts/python.exe Week3/code/collect_results.py --copy-logs
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "Week3" / "configs" / "exp" / "experiments.json"
OUT_MD = ROOT / "Week3" / "deliverables" / "实验结果汇总.md"
LOG_DIR = ROOT / "Week3" / "deliverables" / "logs"

# 归档到 deliverables/logs/ 的文件清单（checkpoint 等大文件不归档）。
# Files archived under deliverables/logs/ (large checkpoints excluded).
LOG_FILES = ["trainer_log.jsonl", "train_results.json", "trainer_state.json",
             "training_loss.png", "run_meta.json", "console.log"]


def load_json(p: Path):
    """容错读 JSON：文件缺失返回 None（实验可能还没跑完）。
    Fault-tolerant JSON load: return None if missing (run may not be done)."""
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def eval_losses(trainer_state: dict) -> tuple:
    """从 log_history 提取 (best_eval_loss, final_eval_loss)。
    Extract (best, final) eval loss from the trainer log history."""
    if not trainer_state:
        return None, None
    evs = [h["eval_loss"] for h in trainer_state.get("log_history", [])
           if "eval_loss" in h]
    return (min(evs), evs[-1]) if evs else (None, None)


def collect_run(exp: dict) -> dict:
    """收集单个实验的全部指标。Collect every metric for one run."""
    out = ROOT / exp["output_dir"]
    tr = load_json(out / "train_results.json")
    ts = load_json(out / "trainer_state.json")
    meta = load_json(out / "run_meta.json")
    best_ev, final_ev = eval_losses(ts)
    return {
        **exp,
        "done": tr is not None,
        "train_loss": tr.get("train_loss") if tr else None,
        "best_eval_loss": best_ev,
        "final_eval_loss": final_ev,
        "wall": meta.get("wall_pretty") if meta else None,
        "peak_vram_gb": round(meta["peak_vram_mib"] / 1024, 1)
        if meta and meta.get("peak_vram_mib") else None,
    }


def fmt(v, nd=4):
    """表格单元格格式化：None 显示为 '—'。Format a cell; None renders as '—'."""
    if v is None:
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def group_rows(rows: list, group: str) -> list:
    """取属于某组的实验（基线的 groups 为 'A+B+C'，三组都含它）。
    Rows belonging to a group (the baseline's 'A+B+C' matches every group)."""
    return [r for r in rows if group in r["groups"]]


def render_model_section(model: str, rows: list) -> list:
    """渲染一个模型的三组对比表。Render the three group tables for one model."""
    title = {"qwen": "Qwen2.5-3B-Instruct", "llama": "Llama-3.2-3B-Instruct"}[model]
    lines = [f"## {title}", ""]
    specs = [("A", "LoRA 秩对比 / rank sweep", "lora_rank"),
             ("B", "学习率对比 / learning-rate sweep", "learning_rate"),
             ("C", "训练轮数对比 / epoch sweep", "num_train_epochs")]
    for g, gtitle, var in specs:
        lines += [f"### 组{g}：{gtitle}", "",
                  "| run_id | 变量值 | train loss | best eval loss | "
                  "final eval loss | 耗时 | 峰值显存 |",
                  "|---|---|---|---|---|---|---|"]
        grows = group_rows(rows, g)
        # 组内最优 = final eval loss 最小者，标 ★（判定规则见实验计划表 §6）。
        # Group winner = lowest final eval loss, marked with ★.
        evs = [r["final_eval_loss"] for r in grows if r["final_eval_loss"]]
        best = min(evs) if evs else None
        for r in grows:
            star = " ★" if best and r["final_eval_loss"] == best else ""
            vram = f"{r['peak_vram_gb']} GB" if r["peak_vram_gb"] else "—"
            lines.append(
                f"| `{r['run_id']}`{star} | {r[var]} | {fmt(r['train_loss'])} | "
                f"{fmt(r['best_eval_loss'])} | {fmt(r['final_eval_loss'])} | "
                f"{fmt(r['wall'])} | {vram} |")
        lines.append("")
    return lines


def copy_logs(rows: list) -> int:
    """把每组的原始日志复制到 deliverables/logs/<run_id>/。
    Copy each run's raw logs into deliverables/logs/<run_id>/."""
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

    exps = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = [collect_run(e) for e in exps]
    done = sum(r["done"] for r in rows)

    lines = [
        f"# Week3 实验结果汇总（{len(rows)} 组 LoRA 超参对比）",
        "",
        "> 本文件由 `Week3/code/collect_results.py` 自动生成，手改会被覆盖；",
        "> 分析结论见《第3周_SFT优化与评估报告.md》。",
        f"> 完成进度 / progress: **{done}/{len(rows)}**。",
        "> ★ = 组内最优（final eval loss 最小，判定规则见《实验计划表》§6）。",
        "",
    ]
    for model in ("qwen", "llama"):
        mrows = [r for r in rows if r["model"] == model]
        if mrows:  # 清单里没有的模型不渲染空段落 / skip models absent from the manifest
            lines += render_model_section(model, mrows)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {OUT_MD.relative_to(ROOT)}  ({done}/{len(rows)} runs done)")

    if args.copy_logs:
        n = copy_logs(rows)
        print(f"[OK] {n} log files -> {LOG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
