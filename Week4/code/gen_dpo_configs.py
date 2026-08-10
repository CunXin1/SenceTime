"""
gen_dpo_configs.py — Week4 Day19
从 template_dpo_qwen.yaml 生成 3 组 DPO 对比实验配置 + experiments.json 清单。
Render 3 DPO comparison configs from the template + an experiments.json manifest.

实验设计（控制变量法，沿用 Week3 方法论）/ Experiment design:
    基线 baseline = β=0.1 / lr=5e-6（任务书指定组，主交付模型）。
    组A 变 β：0.1 vs 0.5（KL 约束强度）。
    组B 变 lr：5e-6 vs 1e-5（学习率）。
    固定：policy=week3-best-merged、r=32/α=64、ep=2、等效 batch 8、cutoff 1024、seed 42。

用法 / Usage（仓库根目录 / from repo root）:
    .venv/Scripts/python.exe Week4/code/gen_dpo_configs.py
"""

import json
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[2]
W4 = ROOT / "Week4"
TEMPLATE = W4 / "configs" / "template_dpo_qwen.yaml"
OUT_DIR = W4 / "configs" / "exp"

# 基线超参 / baseline hyperparameters（任务书指定 β=0.1, lr=5e-6）
BASE = dict(beta="0.1", lr="5.0e-6", ep=2)


def lr_short(lr: str) -> str:
    """'5.0e-6' -> '5e-6'（文件名友好）/ filename-friendly."""
    return lr.replace(".0e", "e")


def build_experiments() -> list[dict]:
    """3 组实验：基线 + 组A(β) 变体 + 组B(lr) 变体。groups 标记对比组。
    3 runs: baseline + group A (β) variant + group B (lr) variant."""
    exps = [
        {**BASE, "groups": "A+B", "note": "baseline 基线（任务书指定组）"},
        {**BASE, "beta": "0.5", "groups": "A", "note": "强 KL 约束 / stronger KL"},
        {**BASE, "lr": "1.0e-5", "groups": "B", "note": "大学习率 / higher lr"},
    ]
    for e in exps:
        # 命名规范 beta{β}_lr{lr}，同时作 saves/week4 子目录名。
        e["name"] = f"beta{e['beta']}_lr{lr_short(e['lr'])}"
    return exps


def render(template_text: str, exp: dict) -> str:
    """填充占位符并剔除 #!TPL 模板注释行（同 Week3 gen_configs.py 的做法）。
    Fill placeholders and strip #!TPL comment lines (same as Week3)."""
    header = (
        f"#  Week4 DPO 实验配置（gen_dpo_configs.py 自动生成，勿手改 / GENERATED — do not edit）\n"
        f"#  实验 / Experiment : qwen_dpo_{exp['name']}   [组/group {exp['groups']}] {exp['note']}\n"
        f"#  变量 / Variables  : pref_beta={exp['beta']}, lr={exp['lr']}, epochs={exp['ep']}\n"
        f"#  运行 / Run        : .venv/Scripts/python.exe Week4/code/run_dpo.py --only {exp['name']}"
    )
    # 先剔除 #!TPL 行再替换（模板注释里的字面 ${...} 会让 Template 报 Invalid placeholder）。
    template_text = "\n".join(
        ln for ln in template_text.splitlines() if not ln.lstrip().startswith("#!TPL"))
    body = Template(template_text).substitute(
        gen_header=header,
        pref_beta=exp["beta"],
        learning_rate=exp["lr"],
        output_dir=f"saves/week4/qwen/qwen_dpo_{exp['name']}",
    )
    return body + ("\n" if not body.endswith("\n") else "")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 重新生成前清掉旧配置，避免清单与 yaml 不同步（同 Week3）。
    for old in OUT_DIR.glob("*.yaml"):
        old.unlink()

    tpl = TEMPLATE.read_text(encoding="utf-8")
    exps = build_experiments()
    manifest = []
    for exp in exps:
        run_id = f"qwen_dpo_{exp['name']}"
        cfg_path = OUT_DIR / f"{run_id}.yaml"
        cfg_path.write_text(render(tpl, exp), encoding="utf-8")
        manifest.append({
            "run_id": run_id,
            "model": "qwen",
            "groups": exp["groups"],
            "pref_beta": exp["beta"],
            "learning_rate": exp["lr"],
            "num_train_epochs": exp["ep"],
            "lora_rank": 32,
            "config": str(cfg_path.relative_to(ROOT)).replace("\\", "/"),
            "output_dir": f"saves/week4/qwen/{run_id}",
        })
        print(f"[OK] {cfg_path.relative_to(ROOT)}")

    mf = OUT_DIR / "experiments.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"\n共生成 {len(manifest)} 个配置 / generated {len(manifest)} configs")
    print(f"清单 / manifest: {mf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
