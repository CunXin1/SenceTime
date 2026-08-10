"""
gen_configs.py — Week3 Day11
批量生成 LoRA 超参对比实验配置。
Batch-generate LoRA hyperparameter comparison configs.

实验设计（控制变量法）/ Experiment design (single-variable control):
    基线 baseline = r8 / lr=1e-4 / ep3，被三组共享、只训练一次。
    The baseline (r8 / lr=1e-4 / ep3) is shared by all three groups.

    默认矩阵 4 组 / default matrix, 4 runs:
        组A 秩   / rank:   8 vs 32（α 同步 = 2r / α follows 2r）
        组B 学习率/ lr:     1e-4 vs 2e-4
        组C 轮数 / epochs: 3 vs 5
    扩展点 / extended points（--full）: r64、lr=5e-5、ep=2

用法 / Usage（在仓库根目录 SenceTime_Week1/ 下运行 / run from repo root）:
    .venv/Scripts/python.exe Week3/code/gen_configs.py                 # 默认：qwen 4 组
    .venv/Scripts/python.exe Week3/code/gen_configs.py --full          # 7 组完整矩阵
    .venv/Scripts/python.exe Week3/code/gen_configs.py --models qwen,llama --full  # 双模型 14 组

输出 / Output:
    Week3/configs/exp/{model}_{exp}.yaml   可运行训练配置 / runnable configs
    Week3/configs/exp/experiments.json     机器可读实验清单，供 run_experiments.py
                                           与 collect_results.py 消费
                                           machine-readable manifest consumed by
                                           run_experiments.py / collect_results.py
"""

import argparse
import json
from pathlib import Path
from string import Template

# 路径基准：本文件位于 Week3/code/，向上两级即仓库根目录。
# Path anchor: this file lives in Week3/code/, two levels up is the repo root.
ROOT = Path(__file__).resolve().parents[2]
W3 = ROOT / "Week3"
OUT_DIR = W3 / "configs" / "exp"

# 两个基座共用同一套实验矩阵（跨架构验证超参结论的泛化性）。
# Both bases share the same experiment matrix (cross-architecture generality).
MODELS = {
    "qwen": W3 / "configs" / "template_qwen.yaml",
    "llama": W3 / "configs" / "template_llama.yaml",
}

# 基线超参 / baseline hyperparameters（Week2 的原始配置 / same as Week2）
BASE = dict(rank=8, lr="1.0e-4", ep=3)


def lr_short(lr: str) -> str:
    """把 '1.0e-4' 压成文件名友好的 '1e-4'。
    Compress '1.0e-4' into the filename-friendly '1e-4'."""
    return lr.replace(".0e", "e")


def build_experiments(full: bool) -> list[dict]:
    """构造实验列表（每模型）：精简 5 个 / 完整 7 个。groups 字段标记对比组。
    Build the experiment list (per model): 5 in lite mode, 7 in full mode.
    The `groups` field marks which comparison group(s) each run belongs to."""
    exps = [
        # 基线被 A/B/C 三组共享，只训练一次 / baseline shared by A+B+C, trained once
        {**BASE, "groups": "A+B+C", "note": "baseline 基线"},
        # 组A：变秩，α=2r / group A: vary rank
        {**BASE, "rank": 32, "groups": "A", "note": "秩加大 / higher rank"},
        # 组B：变学习率 / group B: vary learning rate
        {**BASE, "lr": "2.0e-4", "groups": "B", "note": "大学习率 / higher lr"},
        # 组C：变轮数 / group C: vary epochs
        {**BASE, "ep": 5, "groups": "C", "note": "多轮 / more epochs"},
    ]
    if full:
        exps += [
            {**BASE, "rank": 64, "groups": "A", "note": "秩最大 / highest rank"},
            {**BASE, "lr": "5.0e-5", "groups": "B", "note": "小学习率 / lower lr"},
            {**BASE, "ep": 2, "groups": "C", "note": "少轮 / fewer epochs"},
        ]
    for e in exps:
        # 命名规范 r{rank}_lr{lr}_ep{ep}，同时用作 saves/week3 的子目录名。
        # Naming scheme r{rank}_lr{lr}_ep{ep}; also the saves/week3 sub-dir name.
        e["name"] = f"r{e['rank']}_lr{lr_short(e['lr'])}_ep{e['ep']}"
        e["alpha"] = 2 * e["rank"]  # α=2r 恒定 / keep α/r = 2 constant
    return exps


def render(template_text: str, model: str, exp: dict) -> str:
    """填充模板占位符并剔除模板专用注释行（#!TPL 开头）。
    Fill template placeholders and strip template-only comment lines (#!TPL)."""
    header = (
        f"#  Week3 实验配置（gen_configs.py 自动生成，勿手改 / GENERATED — do not edit）\n"
        f"#  实验 / Experiment : {model}_{exp['name']}   [组/group {exp['groups']}] {exp['note']}\n"
        f"#  变量 / Variables  : lora_rank={exp['rank']} (alpha={exp['alpha']}), "
        f"lr={exp['lr']}, epochs={exp['ep']}\n"
        f"#  运行 / Run        : .venv/Scripts/llamafactory-cli.exe train "
        f"Week3/configs/exp/{model}_{exp['name']}.yaml"
    )
    # 先剔除 #!TPL 行再做替换：模板专用注释里的字面 ${...} 会让 Template 报
    # "Invalid placeholder"。 Strip #!TPL lines BEFORE substitution: the literal
    # ${...} inside template-only comments would otherwise break Template.
    template_text = "\n".join(
        ln for ln in template_text.splitlines() if not ln.lstrip().startswith("#!TPL")
    )
    body = Template(template_text).substitute(
        gen_header=header,
        lora_rank=exp["rank"],
        lora_alpha=exp["alpha"],
        learning_rate=exp["lr"],
        num_train_epochs=f"{exp['ep']}.0",
        output_dir=f"saves/week3/{model}/{exp['name']}",
    )
    return body + ("\n" if not body.endswith("\n") else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="qwen",
                    help="逗号分隔：qwen,llama（默认只 qwen）/ comma-separated model list")
    ap.add_argument("--full", action="store_true",
                    help="完整 7 组矩阵（默认精简 5 组）/ full 7-run matrix per model")
    args = ap.parse_args()
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 重新生成前清掉旧配置，避免清单与 yaml 不同步。
    # Wipe stale configs so yamls never drift from the manifest.
    for old in OUT_DIR.glob("*.yaml"):
        old.unlink()
    exps = build_experiments(args.full)
    manifest = []

    for model, tpl_path in MODELS.items():
        if model not in wanted:
            continue
        tpl = tpl_path.read_text(encoding="utf-8")
        for exp in exps:
            cfg_path = OUT_DIR / f"{model}_{exp['name']}.yaml"
            cfg_path.write_text(render(tpl, model, exp), encoding="utf-8")
            # 清单记录 run_experiments.py / collect_results.py 需要的全部字段。
            # The manifest records everything the runner/collector needs.
            manifest.append({
                "run_id": f"{model}_{exp['name']}",
                "model": model,
                "groups": exp["groups"],
                "lora_rank": exp["rank"],
                "lora_alpha": exp["alpha"],
                "learning_rate": exp["lr"],
                "num_train_epochs": exp["ep"],
                "config": str(cfg_path.relative_to(ROOT)).replace("\\", "/"),
                "output_dir": f"saves/week3/{model}/{exp['name']}",
            })
            print(f"[OK] {cfg_path.relative_to(ROOT)}")

    mf = OUT_DIR / "experiments.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n共生成 {len(manifest)} 个配置 / generated {len(manifest)} configs")
    print(f"清单 / manifest: {mf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
