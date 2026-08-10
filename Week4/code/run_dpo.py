"""
run_dpo.py — Week4 Day19
批量顺序运行 3 组 DPO 对比实验，记录耗时/峰值显存，并对训练时间做硬约束。
Sequentially run the 3 DPO experiments, recording wall-clock time and peak VRAM,
with hard limits on training time.

在 Week3 run_experiments.py 基础上，为回应"训练时间不能太长"新增：
On top of Week3's run_experiments.py, added (to keep training time bounded):
    - EtaMonitor：每 30s 读 trainer_log.jsonl 末行打印进度与预计剩余时间。
      Prints step progress and ETA every 30s from trainer_log.jsonl.
    - --max-minutes N：单组超时即终止该组（run_meta 记 exit_code=timeout）并中止后续。
      Per-run timeout: terminate the run and abort the rest.
    - --budget-min N：累计墙钟超预算则跳过剩余组。
      Total wall-clock budget: skip remaining runs once exceeded.

复用 Week3 的：VramMonitor、make_smoke_config、tee→console.log、run_meta.json、
断点续跑 already_done、--only/--smoke/--dry-run、首个失败即停。

用法 / Usage（仓库根目录 / from repo root）:
    .venv/Scripts/python.exe Week4/code/run_dpo.py --smoke                    # 冒烟（50 条样本）
    .venv/Scripts/python.exe Week4/code/run_dpo.py --max-minutes 40 --budget-min 120
    .venv/Scripts/python.exe Week4/code/run_dpo.py --only beta0.1_lr5e-6      # 只跑基线组
    .venv/Scripts/python.exe Week4/code/run_dpo.py --dry-run
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
MANIFEST = ROOT / "Week4" / "configs" / "exp" / "experiments.json"
LF_CLI = ROOT / ".venv" / "Scripts" / "llamafactory-cli.exe"


class VramMonitor:
    """后台轮询 nvidia-smi 记录峰值显存（MiB）。复用 Week3 run_experiments.py。
    Poll nvidia-smi in a background thread; keep peak VRAM (MiB)."""

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
                capture_output=True, text=True, timeout=10)
            return max(int(x) for x in out.stdout.split())
        except Exception:
            return 0

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


class EtaMonitor:
    """后台每 interval 秒读 trainer_log.jsonl 末行，打印步数进度与预计剩余时间。
    Poll trainer_log.jsonl every `interval` s; print step progress and ETA.
    这是本周对"监控训练时间"约束的直接回应——训练一开始就能看到还要跑多久。
    Directly answers the 'watch the training time' requirement — ETA is visible early."""

    def __init__(self, log_path: Path, run_id: str, interval_s: float = 30.0):
        self.log_path = log_path
        self.run_id = run_id
        self.interval_s = interval_s
        self.eta_at_10pct = None       # 第 ~10% 步时 LF 报的剩余时间 / ETA snapshot near 10%
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _last_record(self) -> dict | None:
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
            for ln in reversed(lines):
                ln = ln.strip()
                if ln:
                    return json.loads(ln)
        except Exception:
            return None
        return None

    def _loop(self):
        while not self._stop.wait(self.interval_s):
            rec = self._last_record()
            if not rec or "current_steps" not in rec:
                continue
            cur, tot = rec.get("current_steps"), rec.get("total_steps")
            pct = rec.get("percentage", 0.0)
            elapsed = rec.get("elapsed_time", "?")
            remaining = rec.get("remaining_time", "?")
            print(f"[eta] {self.run_id} step {cur}/{tot} ({pct:.1f}%) "
                  f"elapsed {elapsed} remaining {remaining}", flush=True)
            # 记录 10% 附近的剩余时间快照（用于周报"预估 vs 实测"）。
            if self.eta_at_10pct is None and pct >= 8.0:
                self.eta_at_10pct = remaining

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


def make_smoke_config(cfg_path: Path, run_id: str) -> Path:
    """冒烟配置：取 50 条样本、输出到 saves/week4/_smoke/。复用 Week3 思路。
    Smoke config: 50 samples, output under saves/week4/_smoke/."""
    text = cfg_path.read_text(encoding="utf-8")
    smoke_out = f"saves/week4/_smoke/{run_id}"
    lines = []
    for ln in text.splitlines():
        if ln.startswith("output_dir:"):
            ln = f"output_dir: {smoke_out}"
        lines.append(ln)
    lines.append("max_samples: 50   # smoke test only")
    smoke_cfg = cfg_path.parent / f"_smoke_{cfg_path.name}"
    smoke_cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return smoke_cfg


def run_one(exp: dict, smoke: bool, max_minutes: float) -> dict:
    """跑一组 DPO 实验并返回 run_meta 字典；支持单组超时终止。
    Run one DPO experiment; supports per-run timeout termination."""
    run_id = exp["run_id"]
    cfg = ROOT / exp["config"]
    if smoke:
        cfg = make_smoke_config(cfg, run_id)
        out_dir = ROOT / "saves" / "week4" / "_smoke" / run_id
    else:
        out_dir = ROOT / exp["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    console_log = out_dir / "console.log"
    trainer_log = out_dir / "trainer_log.jsonl"

    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    timed_out = False
    eta_mon = EtaMonitor(trainer_log, run_id)
    with VramMonitor() as mon, eta_mon, \
            open(console_log, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(
            [str(LF_CLI), "train", str(cfg)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        # 逐行透传 + 落盘；每行都检查是否超单组时限（不阻塞在 read 上太久，
        # DPO 日志较密集，检查粒度足够）。
        # Tee output; check per-run timeout on each line.
        deadline = t0 + max_minutes * 60 if max_minutes > 0 else None
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
            if deadline and time.time() > deadline:
                timed_out = True
                proc.terminate()
                msg = (f"\n!! 单组超时 {max_minutes} 分钟，终止本组 / "
                       f"per-run timeout, terminating {run_id}\n")
                print(msg, end="")
                logf.write(msg)
                break
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    wall_s = round(time.time() - t0, 1)

    exit_code = "timeout" if timed_out else proc.returncode
    meta = {
        "run_id": run_id,
        "config": exp["config"],
        "smoke": smoke,
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": wall_s,
        "wall_pretty": f"{int(wall_s // 60)}m{int(wall_s % 60):02d}s",
        "peak_vram_mib": mon.peak_mib,
        "eta_at_10pct": eta_mon.eta_at_10pct,
        "exit_code": exit_code,
        # 变量快照 / hyperparameter snapshot
        "pref_beta": exp["pref_beta"],
        "learning_rate": exp["learning_rate"],
        "num_train_epochs": exp["num_train_epochs"],
        "lora_rank": exp["lora_rank"],
        "groups": exp["groups"],
        "model": exp["model"],
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def already_done(exp: dict) -> bool:
    """断点续跑判据：train_results.json 存在且 exit_code 为 0。
    Resume criterion: train_results.json exists and exit_code == 0."""
    out_dir = ROOT / exp["output_dir"]
    if not (out_dir / "train_results.json").exists():
        return False
    meta_p = out_dir / "run_meta.json"
    if meta_p.exists():
        try:
            return json.loads(meta_p.read_text(encoding="utf-8"))["exit_code"] == 0
        except Exception:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="只跑 run_id 含此子串的组 / substring filter")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟测试：第一组 + max_samples=50 / smoke test")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划 / print plan only")
    ap.add_argument("--max-minutes", type=float, default=40.0,
                    help="单组训练时限（分钟），超时终止该组，0=不限 / per-run timeout")
    ap.add_argument("--budget-min", type=float, default=120.0,
                    help="总墙钟预算（分钟），超预算跳过剩余组，0=不限 / total budget")
    args = ap.parse_args()

    exps = json.loads(MANIFEST.read_text(encoding="utf-8"))
    exps = [e for e in exps if args.only in e["run_id"]]
    if args.smoke:
        exps = exps[:1]

    todo = [e for e in exps if args.smoke or not already_done(e)]
    skipped = [e["run_id"] for e in exps if e not in todo]
    print(f"计划 / plan: {len(todo)} to run, {len(skipped)} skipped (done): {skipped}")
    print(f"时间约束 / limits: 单组 ≤{args.max_minutes}min, 总预算 ≤{args.budget_min}min")
    if args.dry_run:
        for e in todo:
            print(f"  would run: {e['run_id']}  ({e['config']})")
        return

    results = []
    batch_t0 = time.time()
    for i, exp in enumerate(todo, 1):
        # 总预算闸：开跑前先检查累计墙钟。
        elapsed_min = (time.time() - batch_t0) / 60
        if args.budget_min > 0 and elapsed_min > args.budget_min:
            remaining = [e["run_id"] for e in todo[i - 1:]]
            print(f"\n!! 累计墙钟 {elapsed_min:.1f}min 超总预算 {args.budget_min}min，"
                  f"跳过剩余 {len(remaining)} 组 / budget exceeded, skipping: {remaining}",
                  file=sys.stderr)
            break
        print(f"\n{'=' * 70}\n[{i}/{len(todo)}] {exp['run_id']}  "
              f"(β={exp['pref_beta']}, lr={exp['learning_rate']}, "
              f"ep={exp['num_train_epochs']})\n{'=' * 70}")
        meta = run_one(exp, smoke=args.smoke, max_minutes=args.max_minutes)
        status = "OK" if meta["exit_code"] == 0 else f"FAIL({meta['exit_code']})"
        print(f"\n[{status}] {exp['run_id']}  time={meta['wall_pretty']}  "
              f"peak_vram={meta['peak_vram_mib']}MiB  eta@10%={meta['eta_at_10pct']}")
        results.append(meta)
        # 失败（含超时）即停：后续组大概率同因失败或同样超时。
        if meta["exit_code"] != 0:
            print("!! 训练失败/超时，停止后续实验 / training failed or timed out, aborting",
                  file=sys.stderr)
            sys.exit(1)

    total_min = (time.time() - batch_t0) / 60
    print(f"\n全部完成 / all done: {len(results)} runs, 累计 {total_min:.1f}min")


if __name__ == "__main__":
    main()
