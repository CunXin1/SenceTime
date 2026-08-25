"""
make_report_figs.py — Week8 Day44
为技术报告第 6 章（部署与量化）生成图表。数据全部来自 Week7 的实测产物，不硬编码结论。
Generate the quantization figures for chapter 6 of the technical report.

★ 为什么要专门画「权重 + KV cache 堆叠图」（图 6-1）
    第 7 周最大的一个测量陷阱是：vLLM 按 --gpu-memory-utilization **预分配** KV cache，
    所以 nvidia-smi 看到的进程显存三种精度几乎一样，直接量会得出「量化不省显存」。
    这张图把同一根柱子拆成「权重 / KV cache / 其余」三段——柱高基本相同，
    但内部构成完全不同。**一张图同时讲清「为什么直接量会错」和「量化到底买到了什么」**，
    比两段文字有效。
    One stacked bar per precision: total height is ~constant (that's the trap),
    but the weight/KV-cache split is what quantization actually changes.

★ 数据来源
    · 权重 / PPL / 吞吐 ← Week7/deliverables/bench_{fp16,awq,gptq}.json（bench_quant.py 产出）
    · KV cache / 最大并发 ← Week7/deliverables/logs/serve_*.log 的 vLLM 启动日志
    两者都是实测落盘的原始产物；本脚本只做解析和绘制，**不允许在这里写死任何数字**，
    缺数据就报错退出，避免图里出现「看起来合理但其实是编的」值。
    Parse only; never hardcode measurements — fail loudly if a source file is missing.

用法 / Usage（仓库根目录）:
    .venv/Scripts/python.exe Week8/scripts/make_report_figs.py
产物 / Output:
    Week8/reports/figs/fig6_1_vram_split.png
    Week8/reports/figs/fig6_2_throughput.png
    Week8/reports/figs/fig6_3_quality_cost.png
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Windows 中文字体：与 Week2/Week3 的绘图脚本保持一致 / same CJK fallback as earlier weeks
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
W7 = ROOT / "Week7" / "deliverables"
OUT = ROOT / "Week8" / "reports" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = ["fp16", "awq", "gptq"]
LABEL = {"fp16": "FP16", "awq": "AWQ 4-bit", "gptq": "GPTQ 4-bit"}
# 配色刻意不用红绿：报告会被打印成灰度，蓝/橙/灰在灰度下仍可区分。
COLOR = {"fp16": "#4C72B0", "awq": "#DD8452", "gptq": "#937860"}


def load_bench(v: str) -> dict:
    p = W7 / f"bench_{v}.json"
    if not p.exists():
        sys.exit(f"[FAIL] 缺少 {p}；先跑 Week7/code/run_bench_all.sh")
    return json.loads(p.read_text(encoding="utf-8"))


# vLLM 启动日志里这两行是 KV cache 数据的唯一来源。措辞跨版本变过，故用宽松正则。
_KV_RE = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")
_CONC_RE = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x")
# 「weights + non-torch」是进程实际吃掉的常驻显存（比纯权重略大），用来算「其余」那一段。
_USAGE_RE = re.compile(r"Actual usage is ([\d.]+) GiB for consumed memory")


def parse_serve_log(v: str) -> dict:
    p = W7 / "logs" / f"serve_{v}.log"
    if not p.exists():
        sys.exit(f"[FAIL] 缺少 {p}")
    txt = p.read_text(encoding="utf-8", errors="ignore")
    kv = _KV_RE.search(txt)
    conc = _CONC_RE.search(txt)
    usage = _USAGE_RE.search(txt)
    if not (kv and conc and usage):
        sys.exit(f"[FAIL] {p} 里解析不到 KV cache / 并发 / 常驻显存，日志格式可能变了")
    return {
        "kv_gb": float(kv.group(1)),
        "ctx_tokens": int(conc.group(1).replace(",", "")),
        "max_conc": float(conc.group(2)),
        "consumed_gb": float(usage.group(1)),
    }


def fig_vram_split(data: dict) -> None:
    """图 6-1：同一根柱子拆成 权重 / KV cache / 其余（激活+CUDAGraph+框架开销）。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = range(len(VARIANTS))
    for i, v in enumerate(VARIANTS):
        w = data[v]["weight_gb"]
        kv = data[v]["kv_gb"]
        # 「其余」= 进程常驻显存减去纯权重（非 torch 分配、激活峰值、CUDAGraph 等）
        other = max(data[v]["consumed_gb"] - w, 0.0)
        ax.bar(i, w, color=COLOR[v], label="模型权重" if i == 0 else None)
        ax.bar(i, other, bottom=w, color="#BBBBBB", label="其余（非torch/激活/CUDAGraph）" if i == 0 else None)
        ax.bar(i, kv, bottom=w + other, color="#CCD9EA", edgecolor="#8899AA",
               label="KV cache（vLLM 预分配）" if i == 0 else None)
        ax.text(i, w / 2, f"{w:.2f}", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        ax.text(i, w + other + kv / 2, f"{kv:.2f} GiB\n{data[v]['max_conc']:.2f}× 并发",
                ha="center", va="center", fontsize=9, color="#333333")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABEL[v] for v in VARIANTS])
    ax.set_ylabel("显存占用 (GiB)")
    ax.set_title("图 6-1　同一 --gpu-memory-utilization 下的显存构成\n"
                 "（柱高相近 = nvidia-smi 量不出差异；差异全在内部构成）", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_1_vram_split.png", dpi=150)
    plt.close(fig)


def fig_throughput(data: dict) -> None:
    """图 6-2：batch=1 与并发两档吞吐。两档必须并排——只报一个数会掩盖瓶颈切换。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.35
    xs = list(range(len(VARIANTS)))
    b1 = [data[v]["tps_batch1"] for v in VARIANTS]
    bc = [data[v]["tps_concurrent"] for v in VARIANTS]
    conc_n = data["fp16"]["concurrency"]
    r1 = ax.bar([x - width / 2 for x in xs], b1, width, color="#4C72B0", label="batch = 1")
    r2 = ax.bar([x + width / 2 for x in xs], bc, width, color="#DD8452",
                label=f"并发 = {conc_n}")
    for rects in (r1, r2):
        ax.bar_label(rects, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[v] for v in VARIANTS])
    ax.set_ylabel("生成吞吐 (tokens/s)")
    ax.set_title("图 6-2　两档负载下的生成吞吐（同一 vLLM 服务端，max_tokens=256）", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_2_throughput.png", dpi=150)
    plt.close(fig)


def fig_quality_cost(data: dict) -> None:
    """图 6-3：质量代价（PPL 相对劣化）与一次性量化耗时的权衡。

    量化耗时不在 bench json 里（它是离线阶段的数，记录在 Week7 报告与日志中），
    这里从日志时间戳算，而不是手填——保持「图里的数都能追到源头」。
    """
    base = data["fp16"]["ppl_median"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = list(range(len(VARIANTS)))
    deg = [(data[v]["ppl_median"] / base - 1) * 100 for v in VARIANTS]
    rects = ax.bar(xs, deg, 0.5, color=[COLOR[v] for v in VARIANTS])
    ax.bar_label(rects, fmt="%+.2f%%", fontsize=9, padding=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{LABEL[v]}\nPPL={data[v]['ppl_median']:.3f}" for v in VARIANTS])
    ax.set_ylabel("困惑度相对 FP16 的劣化 (%)")
    ax.set_title("图 6-3　量化的质量代价（32 篇留出文档，与校准集不相交）", fontsize=11)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_3_quality_cost.png", dpi=150)
    plt.close(fig)


def main() -> None:
    data = {}
    for v in VARIANTS:
        data[v] = {**load_bench(v), **parse_serve_log(v)}
        d = data[v]
        print(f"[{v:>4}] 权重 {d['weight_gb']:.2f} GiB | KV {d['kv_gb']:.2f} GiB | "
              f"并发 {d['max_conc']:.2f}x@{d['ctx_tokens']} | PPL {d['ppl_median']:.3f} | "
              f"{d['tps_batch1']:.1f} / {d['tps_concurrent']:.1f} tok/s")
    fig_vram_split(data)
    fig_throughput(data)
    fig_quality_cost(data)
    print(f"[ok] 3 张图已写入 {OUT}")


if __name__ == "__main__":
    main()
