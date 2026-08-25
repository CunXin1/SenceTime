"""merge_model.py — Week8 Day40 / 任务书 40.3「训练完成后的模型合并脚本」

把 LoRA adapter 合并回基座，产出一个可以直接部署的完整模型目录。
本质是对 `python -m llamafactory.cli export` 的一层封装，但把"合并前该检查
什么"固化了下来——流水线里合并是**无人值守**跑的，出问题时没人在旁边看日志。
A guarded wrapper around LLaMA-Factory's export; the guards matter because the
merge runs unattended inside the pipeline.

--------------------------------------------------------------------------
★ 取舍一：为什么必须是 `python -m llamafactory.cli export`，不能用 llamafactory-cli.exe
    Week2 Day10 实测：Windows 上走 .venv/Scripts/llamafactory-cli.exe 这个
    console_script wrapper 会**段错误**（wrapper 自身的进程初始化与 torch 的
    CUDA DLL 加载顺序冲突，崩在 python 解释器之外，连 traceback 都拿不到）。
    `python -m` 直接由解释器加载模块，绕开 wrapper。这条是硬约束，本脚本不提供
    走 exe 的选项 —— 留一个会崩的选项等于给未来的自己埋雷。
    On Windows the console_script entry point segfaults; always use `python -m`.

★ 取舍二：为什么合并前一定要查磁盘，而不是"跑挂了再说"
    一个 3B bf16 模型落盘约 6GB，本机 C 盘长期在 90%+（写这个脚本时剩 35GB）。
    磁盘写满时 safetensors 的失败方式很难看：export 已经写了几个分片才 ENOSPC，
    留下一个**看起来存在、实际残缺**的模型目录。下一步（DPO 训练 / 部署）会把它
    当成正常模型加载，报出来的错是 "missing key ..." 之类，要追很久才想到是磁盘。
    先查余量、不够就早失败，是把一个隐蔽故障换成一个明确故障。
    Pre-flight the disk: a half-written model dir fails in confusing ways later.

★ 取舍三：目标目录已存在时默认**拒绝**而不是覆盖
    合并耗时几分钟，覆盖掉一个已经被 step3 评测过、或者已经被量化过的模型，
    代价远大于重跑一次。所以默认报错退出，要覆盖必须显式 --force。
    流水线里 step2_train.sh 传 --force（它明确知道自己在重跑整条链）。

用法 / Usage:
    # 方式一（流水线用）：直接跑一份现成的合并配置
    .venv/Scripts/python.exe Week8/scripts/merge_model.py --config Week8/configs/merge_sft.yaml

    # 方式二：全参数化，脚本自己生成临时配置
    .venv/Scripts/python.exe Week8/scripts/merge_model.py \
        --base models/Qwen2.5-3B-Instruct \
        --adapter saves/week8/qwen/sft_best \
        --export-dir models/Qwen2.5-3B-week8-sft-merged \
        --template qwen

    # 只看要执行什么，不真跑
    .venv/Scripts/python.exe Week8/scripts/merge_model.py --config ... --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# adapter 目录里至少要有这两个之一，否则 LF 会在加载到一半时才报错。
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def rp(p) -> Path:
    """相对路径一律按仓库根解析 —— 本脚本可能被 step2_train.sh 从任意 cwd 调用。"""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 ** 3)


def free_gb(path: Path) -> float:
    """目标盘余量。取最近一个已存在的祖先目录 —— export_dir 本身通常还不存在，
    对不存在的路径调 disk_usage 会抛 FileNotFoundError。"""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / (1024 ** 3)


def read_yaml_field(cfg_text: str, key: str):
    """从合并配置里抠出一个顶层标量字段。

    ★ 为什么手写而不是 import yaml：本脚本要在"检查阶段"就读到 base/adapter/
      export_dir 三个路径，而此时还不想把 PyYAML（进而可能是整条 LF 依赖链）
      拉进来 —— 检查失败时应该秒退，不该先花十几秒 import torch。
      合并配置是我们自己写的、结构固定的扁平 YAML（无嵌套、无锚点、无多行标量），
      按行取 `key: value` 完全够用。真遇到复杂 YAML 会取不到值，
      那时会走到下面的 "字段缺失" 分支报错，不会静默用错值。
    """
    for line in cfg_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or ":" not in stripped:
            continue
        k, _, v = stripped.partition(":")
        if k.strip() != key:
            continue
        v = v.split("#", 1)[0].strip().strip('"').strip("'")   # 去行尾注释和引号
        return v or None
    return None


def build_temp_config(args) -> Path:
    """方式二：把命令行参数写成一份临时合并配置。
    字段与 merge_sft.yaml / merge_dpo.yaml 完全一致，保证两条路径行为相同。"""
    text = "\n".join([
        "# 由 merge_model.py 自动生成的临时合并配置（可安全删除）",
        f"model_name_or_path: {args.base}",
        f"adapter_name_or_path: {args.adapter}",
        f"template: {args.template}",
        "finetuning_type: lora",
        "trust_remote_code: true",
        f"export_dir: {args.export_dir}",
        f"export_size: {args.export_size}",
        f"export_device: {args.export_device}",
        "export_legacy_format: false",
        "",
    ])
    fd = tempfile.NamedTemporaryFile("w", suffix="_merge.yaml", delete=False,
                                     encoding="utf-8", dir=str(ROOT / "Week8" / "logs"))
    fd.write(text)
    fd.close()
    return Path(fd.name)


def preflight(base: str, adapter: str, export_dir: str, force: bool) -> None:
    """合并前的全部检查。任何一条不过就退出 —— 无人值守时早失败优于晚失败。"""
    ok = True

    base_p = rp(base)
    if not base_p.exists():
        print(f"[FAIL] 基座模型不存在: {base_p}")
        ok = False

    adapter_p = rp(adapter)
    if not adapter_p.exists():
        print(f"[FAIL] adapter 目录不存在: {adapter_p}")
        print("       训练还没跑完，或 output_dir 与本配置的 adapter_name_or_path 对不上。")
        ok = False
    else:
        if not (adapter_p / "adapter_config.json").exists():
            print(f"[FAIL] {adapter_p} 里没有 adapter_config.json，这不是一个 LoRA adapter 目录。")
            ok = False
        if not any((adapter_p / f).exists() for f in ADAPTER_WEIGHT_FILES):
            print(f"[FAIL] {adapter_p} 里找不到 {' 或 '.join(ADAPTER_WEIGHT_FILES)}；"
                  f"训练可能中途崩了，只留下了配置文件。")
            ok = False

    out_p = rp(export_dir)
    if out_p.exists() and any(out_p.iterdir()):
        if force:
            print(f"[warn] 目标目录已存在且非空，--force 生效，将被覆盖: {out_p}")
        else:
            print(f"[FAIL] 目标目录已存在且非空: {out_p}")
            print("       合并会覆盖它。确认要覆盖请加 --force；否则换一个 --export-dir。")
            ok = False

    # 磁盘余量：需要 ≈ 基座大小 × 1.1（合并后参数量不变，留 10% 给 tokenizer/config
    # 和分片切分时的临时占用）。
    if base_p.exists():
        need = dir_size_gb(base_p) * 1.1
        have = free_gb(out_p)
        print(f"[check] 磁盘：需要 ≈{need:.1f}GB，目标盘可用 {have:.1f}GB")
        if have < need:
            print(f"[FAIL] 磁盘余量不足。写到一半 ENOSPC 会留下一个残缺但看起来存在的"
                  f"模型目录，后面很难排查（见文件头 ★取舍二）。")
            ok = False

    if not ok:
        sys.exit(1)
    print("[check] 前置检查全部通过")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="把 LoRA adapter 合并回基座（封装 llamafactory.cli export）")
    ap.add_argument("--config", help="现成的合并配置（如 Week8/configs/merge_sft.yaml）")
    ap.add_argument("--base", help="基座 / policy 模型路径（不用 --config 时必填）")
    ap.add_argument("--adapter", help="LoRA adapter 目录（不用 --config 时必填）")
    ap.add_argument("--export-dir", help="导出目录（不用 --config 时必填）")
    ap.add_argument("--template", default="qwen", help="对话模板，必须与训练时一致")
    ap.add_argument("--export-size", type=int, default=5, help="safetensors 分片上限(GB)")
    ap.add_argument("--export-device", default="cpu",
                    help="cpu / auto；默认 cpu，合并是纯加法不需要显存")
    ap.add_argument("--python", default=sys.executable,
                    help="用哪个解释器跑 export；默认就是当前解释器")
    ap.add_argument("--force", action="store_true", help="目标目录已存在时允许覆盖")
    ap.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    args = ap.parse_args()

    tmp_cfg = None
    if args.config:
        cfg = rp(args.config)
        if not cfg.exists():
            sys.exit(f"[FAIL] 找不到配置 {cfg}")
        text = cfg.read_text(encoding="utf-8")
        base = read_yaml_field(text, "model_name_or_path")
        adapter = read_yaml_field(text, "adapter_name_or_path")
        export_dir = read_yaml_field(text, "export_dir")
        missing = [k for k, v in (("model_name_or_path", base),
                                  ("adapter_name_or_path", adapter),
                                  ("export_dir", export_dir)) if not v]
        if missing:
            sys.exit(f"[FAIL] {cfg} 里缺少字段: {', '.join(missing)}")
    else:
        if not (args.base and args.adapter and args.export_dir):
            sys.exit("[FAIL] 不用 --config 时，--base / --adapter / --export-dir 三个都必填")
        base, adapter, export_dir = args.base, args.adapter, args.export_dir
        tmp_cfg = build_temp_config(args)
        cfg = tmp_cfg

    print(f"[merge] base      : {base}")
    print(f"[merge] adapter   : {adapter}")
    print(f"[merge] export_dir: {export_dir}")
    print(f"[merge] config    : {cfg}")

    # ★ 必须 python -m，见文件头 ★取舍一。
    cmd = [args.python, "-m", "llamafactory.cli", "export", str(cfg)]

    if args.dry_run:
        print("[dry-run] 前置检查（不含覆盖判定）与命令如下，不执行：")
        print("[dry-run] " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return 0

    preflight(base, adapter, export_dir, args.force)

    print("[merge] 开始合并（CPU 上跑，3B 模型约几分钟）...")
    # ★ cwd 固定为仓库根：合并配置里的路径全是仓库相对路径（与 Week3/Week4 一致），
    #   从别处调用时不 cd 过去会全部找不到。
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        print(f"[FAIL] export 退出码 {rc}")
        return rc

    out_p = rp(export_dir)
    print(f"[OK] 合并完成: {out_p}  ({dir_size_gb(out_p):.2f} GB)")
    if tmp_cfg and tmp_cfg.exists():
        tmp_cfg.unlink()          # 临时配置用完即删，不污染 logs 目录
    return 0


if __name__ == "__main__":
    sys.exit(main())
