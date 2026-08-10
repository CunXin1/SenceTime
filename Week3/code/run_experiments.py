"""
run_experiments.py — Week3 Day12-13
批量顺序运行 14 组 LoRA 超参对比实验，并记录耗时与峰值显存。
Sequentially run the 14 LoRA hyperparameter experiments, recording wall-clock
time and peak VRAM for each run.

功能 / Features:
    1. 读取 gen_configs.py 生成的清单 Week3/configs/exp/experiments.json。
       Read the manifest emitted by gen_configs.py.
    2. 逐组调用 llamafactory-cli train（单卡顺序跑，避免显存互相挤占）。
       Invoke llamafactory-cli train one run at a time (single GPU).
    3. 后台线程每 2s 轮询 nvidia-smi，记录本次训练的峰值显存。
       A background thread polls nvidia-smi every 2s to capture peak VRAM.
    4. 每组结束写 {output_dir}/run_meta.json（耗时/显存/退出码/时间戳）。
       Write {output_dir}/run_meta.json after each run.
    5. 断点续跑：已存在 train_results.json 且退出码为 0 的组自动跳过。
       Resumable: runs whose train_results.json already exists are skipped.
    6. 控制台输出同时落盘到 {output_dir}/console.log，作为原始日志交付的一部分。
       Console output is teed into {output_dir}/console.log (raw-log deliverable).

用法 / Usage（在仓库根目录 SenceTime_Week1/ 下 / from repo root）:
    .venv/Scripts/python.exe Week3/code/run_experiments.py                # 全部 14 组 / all runs
    .venv/Scripts/python.exe Week3/code/run_experiments.py --only qwen    # 只跑名字含 qwen 的组
    .venv/Scripts/python.exe Week3/code/run_experiments.py --smoke        # 冒烟测试（100 条样本）
    .venv/Scripts/python.exe Week3/code/run_experiments.py --dry-run      # 只打印将执行什么
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "Week3" / "configs" / "exp" / "experiments.json"
LF_CLI = ROOT / ".venv" / "Scripts" / "llamafactory-cli.exe"


class VramMonitor:
    """后台轮询 nvidia-smi 记录峰值显存（MiB）。
    Poll nvidia-smi in a background thread and keep the peak VRAM (MiB).

    注意：nvidia-smi 统计的是整卡已用显存（含其他进程）。实验期间保持 GPU
    独占即可视为本次训练的峰值。
    Note: nvidia-smi reports card-wide usage (incl. other processes). Keep the
    GPU exclusive during experiments so the peak reflects this run."""

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _query(self) -> int:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            return max(int(x) for x in out.stdout.split())
        except Exception:
            return 0  # nvidia-smi 偶发失败不影响训练 / tolerate transient failures

    def _loop(self):
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, self._query())
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


def make_smoke_config(cfg_path: Path, run_id: str) -> Path:
    """生成冒烟测试配置：取 100 条样本、输出到 saves/week3/_smoke/。
    Build a smoke-test config: 100 samples, output under saves/week3/_smoke/.
    用于全量跑之前验证管线（数据注册/模板/显存记录）是否通畅。
    Used to validate the pipeline (dataset registry / template / VRAM logging)
    before committing ~4.5h of GPU time."""
    text = cfg_path.read_text(encoding="utf-8")
    smoke_out = f"saves/week3/_smoke/{run_id}"
    lines = []
    for ln in text.splitlines():
        if ln.startswith("output_dir:"):
            ln = f"output_dir: {smoke_out}"
        lines.append(ln)
    # 追加 max_samples 覆盖（yaml 后写的键覆盖前面的值不成立，故直接追加新键——
    # max_samples 在模板中不存在，追加即可生效）。
    # Append max_samples (the key does not exist in the template, so appending
    # simply defines it).
    lines.append("max_samples: 100   # smoke test only")
    smoke_cfg = cfg_path.parent / f"_smoke_{cfg_path.name}"
    smoke_cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return smoke_cfg


def run_one(exp: dict, smoke: bool) -> dict:
    """跑一组实验并返回 run_meta 字典。Run one experiment, return its meta."""
    run_id = exp["run_id"]
    cfg = ROOT / exp["config"]
    if smoke:
        cfg = make_smoke_config(cfg, run_id)
        out_dir = ROOT / "saves" / "week3" / "_smoke" / run_id
    else:
        out_dir = ROOT / exp["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    console_log = out_dir / "console.log"

    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    with VramMonitor() as mon, open(console_log, "w", encoding="utf-8",
                                    errors="replace") as logf:
        # 训练输出逐行透传到终端 + 落盘，方便边跑边看且保留原始日志。
        # Tee training output to both terminal and file.
        proc = subprocess.Popen(
            [str(LF_CLI), "train", str(cfg)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        proc.wait()
    wall_s = round(time.time() - t0, 1)

    meta = {
        "run_id": run_id,
        "config": exp["config"],
        "smoke": smoke,
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": wall_s,
        "wall_pretty": f"{int(wall_s // 60)}m{int(wall_s % 60):02d}s",
        "peak_vram_mib": mon.peak_mib,
        "exit_code": proc.returncode,
        # 变量快照，收集脚本无需再解析 yaml / hyperparameter snapshot
        "lora_rank": exp["lora_rank"],
        "learning_rate": exp["learning_rate"],
        "num_train_epochs": exp["num_train_epochs"],
        "groups": exp["groups"],
        "model": exp["model"],
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def already_done(exp: dict) -> bool:
    """断点续跑判据：train_results.json 存在且 run_meta 退出码为 0。
    Resume criterion: train_results.json exists and run_meta exit code is 0."""
    out_dir = ROOT / exp["output_dir"]
    if not (out_dir / "train_results.json").exists():
        return False
    meta_p = out_dir / "run_meta.json"
    if meta_p.exists():
        try:
            return json.loads(meta_p.read_text(encoding="utf-8"))["exit_code"] == 0
        except Exception:
            return False
    return True  # 无 meta 但有结果（手动跑的）也视为完成 / manual run counts too


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="只跑 run_id 含此子串的组 / filter by substring")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟测试：第一组匹配实验 + max_samples=100 / smoke test")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划 / print plan only")
    args = ap.parse_args()

    exps = json.loads(MANIFEST.read_text(encoding="utf-8"))
    exps = [e for e in exps if args.only in e["run_id"]]
    if args.smoke:
        exps = exps[:1]  # 冒烟只跑一组 / smoke runs a single experiment

    todo = [e for e in exps if args.smoke or not already_done(e)]
    skipped = [e["run_id"] for e in exps if e not in todo]
    print(f"计划 / plan: {len(todo)} to run, {len(skipped)} skipped (done): {skipped}")
    if args.dry_run:
        for e in todo:
            print(f"  would run: {e['run_id']}  ({e['config']})")
        return

    results = []
    for i, exp in enumerate(todo, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(todo)}] {exp['run_id']}  "
              f"(r={exp['lora_rank']}, lr={exp['learning_rate']}, "
              f"ep={exp['num_train_epochs']})\n{'=' * 70}")
        meta = run_one(exp, smoke=args.smoke)
        status = "OK" if meta["exit_code"] == 0 else f"FAIL({meta['exit_code']})"
        print(f"\n[{status}] {exp['run_id']}  time={meta['wall_pretty']}  "
              f"peak_vram={meta['peak_vram_mib']}MiB")
        results.append(meta)
        # 一组失败立即停：后续组大概率同因失败，别浪费 GPU 时间。
        # Stop on first failure: later runs would likely fail for the same reason.
        if meta["exit_code"] != 0:
            print("!! 训练失败，停止后续实验 / training failed, aborting the rest",
                  file=sys.stderr)
            sys.exit(1)

    print(f"\n全部完成 / all done: {len(results)} runs")


if __name__ == "__main__":
    main()
