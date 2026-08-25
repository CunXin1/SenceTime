"""
bench_quant.py — Week7 Day35
对同一个 vLLM 服务测三件事：权重显存、困惑度、吞吐。三种精度分别跑一次，产出对比表。
Measure weight VRAM, perplexity and throughput against a running vLLM server.

★ 显存：为什么不能用 nvidia-smi（这是本周最容易出错的一个测量）
    vLLM 启动时按 --gpu-memory-utilization **预分配** KV cache，把显存一次性吃满。
    FP16 和 4-bit 权重差出来的那 4GB，会被 vLLM 拿去多分配 4GB 的 KV cache——
    nvidia-smi 看到的进程显存**三种精度完全一样**。用它量会直接得出"量化不省显存"
    的错误结论，而验收标准❶要的正是显存降幅。
    正确口径是模型**权重**占用，vLLM 在启动日志里会打出来，本脚本解析该日志。
    量化真正买到的不是"更小的进程"，而是"同样的显存里能塞下更多 KV cache /
    更长上下文 / 更高并发"——这一点值得写进报告。
    vLLM pre-allocates KV cache to the utilization ratio, so nvidia-smi shows the
    same total for every precision. Parse the weight line from the startup log.

★ 困惑度：为什么走 vLLM 的 prompt_logprobs 而不是另开 HF
    要比的是"部署态"的质量。若 PPL 用 HF transformers 算、吞吐用 vLLM 测，
    两者的 kernel、反量化实现、dtype 提升策略都不同，PPL 的差异里就掺进了
    "两套实现的差异"。全部走同一个服务端接口，唯一变量才是量化算法本身。
    做法：/v1/completions 传 prompt_logprobs=0，vLLM 会返回 prompt 里每个 token
    的对数概率，累加求平均再取 exp 就是 PPL。max_tokens=1 是因为我们不需要续写。
    Same server, same kernels — the only variable left is the quantization method.

★ 吞吐：为什么必须分 batch=1 和并发两组
    4-bit 在 **batch=1** 是显存带宽瓶颈：权重小一半 → 每步搬的字节少一半 → 明显更快。
    但并发拉高后瓶颈转向算力，而 4-bit 每次矩阵乘前要先反量化回 fp16，这份额外开销
    在大 batch 下摊不掉，吞吐可能**反而低于 FP16**。只报一个数字会掩盖这个拐点。
    Low batch = bandwidth-bound (quantization wins); high batch = compute-bound
    (dequant overhead can make 4-bit slower). Report both.

用法 / Usage（WSL 的 vllm 环境，先在另一个终端起好服务）:
    source ~/venvs/vllm/bin/activate
    python Week7/code/bench_quant.py --variant awq
    python Week7/code/bench_quant.py --variant awq --port 8000 --n-ppl 32 --concurrency 16
产物 / Output:
    Week7/deliverables/bench_<variant>.json   ← 三种精度各跑一次后由 --report 汇总成表
    python Week7/code/bench_quant.py --report
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
PPL_SET = ROOT / "Week7" / "data" / "ppl_eval.json"
DELIV = ROOT / "Week7" / "deliverables"
LOGDIR = DELIV / "logs"

# vLLM 各版本对这行日志的措辞改过好几次，多个模式都试一遍。
# 例："Model loading took 5.7871 GiB and 12.3 seconds"
#     "the model weights take 5.79GiB; ..."  /  "Loading model weights took 5.7871 GB"
WEIGHT_PATTERNS = [
    r"[Mm]odel loading took\s+([\d.]+)\s*GiB",
    r"model weights take\s+([\d.]+)\s*GiB",
    r"Loading model weights took\s+([\d.]+)\s*GB",
    r"model weights.*?([\d.]+)\s*GiB",
]


def parse_weight_gb(variant: str) -> float | None:
    """从 serve_vllm.sh 留下的启动日志里解析模型权重显存。"""
    log = LOGDIR / f"serve_{variant}.log"
    if not log.exists():
        print(f"[warn] 找不到 {log.name}，权重显存留空。先用 serve_vllm.sh 起服务（它会 tee 日志）。")
        return None
    text = log.read_text(encoding="utf-8", errors="ignore")
    for pat in WEIGHT_PATTERNS:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    print(f"[warn] {log.name} 里没匹配到权重占用行，vLLM 措辞可能又变了。"
          f"手动 grep 一下 'GiB' 并补进 WEIGHT_PATTERNS。")
    return None


def measure_ppl(base: str, model: str, n: int, maxlen: int) -> dict:
    """用 prompt_logprobs 算困惑度。

    返回逐文档 PPL 的中位数与均值。取中位数是因为个别长文档里如果混了代码块或
    英文段落，PPL 会有几倍的离群值，均值会被单条样本主导。
    """
    docs = json.loads(PPL_SET.read_text(encoding="utf-8"))[:n]
    ppls: list[float] = []
    with httpx.Client(timeout=300.0) as cli:
        for i, d in enumerate(docs, 1):
            r = cli.post(f"{base}/v1/completions", json={
                "model": model,
                "prompt": d["text"],
                "max_tokens": 1,
                "temperature": 0.0,
                "prompt_logprobs": 0,   # ← 关键：让 vLLM 回传 prompt 每个 token 的 logprob
            })
            r.raise_for_status()
            plp = r.json()["choices"][0].get("prompt_logprobs")
            if not plp:
                raise SystemExit("[FAIL] 服务没返回 prompt_logprobs，检查 vLLM 版本是否支持该参数")
            # plp[0] 是 None（第一个 token 没有条件概率可言），逐位取该 token 自身的 logprob
            lps = []
            for entry in plp[1:]:
                if not entry:
                    continue
                lps.append(max(v["logprob"] for v in entry.values()) if len(entry) == 1
                           else list(entry.values())[0]["logprob"])
            if lps:
                ppls.append(math.exp(-sum(lps) / len(lps)))
            if i % 10 == 0:
                print(f"  [ppl] {i}/{len(docs)} 中位={statistics.median(ppls):.3f}")
    return {"ppl_median": statistics.median(ppls), "ppl_mean": statistics.fmean(ppls),
            "ppl_n": len(ppls)}


PROMPTS = [
    "用三句话解释什么是模型量化，面向没有深度学习背景的同学。",
    "鸡兔同笼：共 35 个头、94 只脚，问鸡兔各几只？给出完整推理。",
    "写一个 Python 函数，判断一个字符串是否是回文，并说明时间复杂度。",
    "把下面这句话翻译成英文并解释语气差异：这件事我们再议。",
]


def one_request(cli: httpx.Client, base: str, model: str, prompt: str, max_tokens: int) -> tuple[int, float]:
    t0 = time.time()
    r = cli.post(f"{base}/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,    # 贪心，与 Week3/4/6 的评测口径一致，保证可复现
    })
    r.raise_for_status()
    j = r.json()
    return j["usage"]["completion_tokens"], time.time() - t0


def measure_throughput(base: str, model: str, rounds: int, concurrency: int, max_tokens: int) -> dict:
    """两组：batch=1 顺序发，以及 concurrency 路并发。"""
    with httpx.Client(timeout=600.0) as cli:
        # --- 组一：batch=1，衡量单用户体感延迟（显存带宽瓶颈区） ---
        toks, secs = 0, 0.0
        for i in range(rounds):
            n, dt = one_request(cli, base, model, PROMPTS[i % len(PROMPTS)], max_tokens)
            toks += n
            secs += dt
        single = toks / secs

        # --- 组二：并发，衡量服务吞吐上限（算力瓶颈区，反量化开销在这里显形） ---
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(one_request, cli, base, model,
                              PROMPTS[i % len(PROMPTS)], max_tokens)
                    for i in range(concurrency)]
            total = sum(f.result()[0] for f in futs)
        batched = total / (time.time() - t0)

    return {"tps_batch1": round(single, 2), "tps_concurrent": round(batched, 2),
            "concurrency": concurrency, "max_tokens": max_tokens}


def report() -> None:
    """把三份 bench_*.json 汇成 Markdown 表（Day35 交付物）。"""
    rows = []
    for v in ("fp16", "awq", "gptq"):
        p = DELIV / f"bench_{v}.json"
        if p.exists():
            rows.append((v, json.loads(p.read_text(encoding="utf-8"))))
    if not rows:
        raise SystemExit("[FAIL] 一份 bench_*.json 都没有，先分别跑三种精度。")

    base_w = dict(rows).get("fp16", {}).get("weight_gb")
    base_p = dict(rows).get("fp16", {}).get("ppl_median")
    lines = [
        "| 精度 | 权重显存 (GiB) | 相对 FP16 降幅 | 困惑度 (中位) | 相对 FP16 | tokens/s (batch=1) | tokens/s (并发) |",
        "|---|---|---|---|---|---|---|",
    ]
    for v, d in rows:
        w, p = d.get("weight_gb"), d.get("ppl_median")
        drop = f"{100 * (1 - w / base_w):.1f}%" if (w and base_w) else "—"
        pdel = f"+{100 * (p / base_p - 1):.2f}%" if (p and base_p) else "—"
        lines.append(f"| {v.upper()} | {w if w else '—'} | {drop} | "
                     f"{f'{p:.3f}' if p else '—'} | {pdel} | "
                     f"{d.get('tps_batch1', '—')} | {d.get('tps_concurrent', '—')} |")
    out = DELIV / "量化对比表.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[ok] -> {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["fp16", "awq", "gptq"],
                    help="当前服务跑的是哪种精度（用于命名产物、定位启动日志）")
    ap.add_argument("--base", default="http://127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen3b", help="serve_vllm.sh 里的 --served-model-name")
    ap.add_argument("--n-ppl", type=int, default=32)
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--report", action="store_true", help="只汇总已有结果成表，不测")
    args = ap.parse_args()

    if args.report:
        report()
        return
    if not args.variant:
        raise SystemExit("需要 --variant {fp16|awq|gptq}（或用 --report 汇总）")

    base = f"{args.base}:{args.port}"
    print(f"[bench] variant={args.variant}  {base}  model={args.model}")

    result: dict = {"variant": args.variant, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    result["weight_gb"] = parse_weight_gb(args.variant)
    print(f"[1/3] 权重显存: {result['weight_gb']} GiB")

    print(f"[2/3] 困惑度（{args.n_ppl} 篇留出文档，与校准集不相交）...")
    result.update(measure_ppl(base, args.model, args.n_ppl, args.maxlen))
    print(f"      PPL 中位={result['ppl_median']:.4f} 均值={result['ppl_mean']:.4f}")

    print(f"[3/3] 吞吐（batch=1 × {args.rounds} 轮，并发 {args.concurrency}）...")
    result.update(measure_throughput(base, args.model, args.rounds,
                                     args.concurrency, args.max_tokens))
    print(f"      batch=1 {result['tps_batch1']} tok/s | "
          f"并发{args.concurrency} {result['tps_concurrent']} tok/s")

    DELIV.mkdir(parents=True, exist_ok=True)
    out = DELIV / f"bench_{args.variant}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[ok] -> {out.relative_to(ROOT)}   三种都跑完后： --report 汇总成表")


if __name__ == "__main__":
    main()
