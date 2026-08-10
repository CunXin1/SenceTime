"""Day7：数据清洗 Pipeline。可独立运行（验收①）。

清洗步骤（顺序即漏斗）：
  1) 去 HTML 标签
  2) 去不可见控制字符 / 零宽字符 / 规范空白
  3) 空值过滤（instruction 或 output 为空则丢）
  4) 超长处理：全文 > MAX_TOKENS(2048) 时截断 output 尾部；instruction 本身超长则丢
  5) 模糊去重：SimHash(64bit) + LSH 分桶 + Hamming 距离阈值（可选 Jaccard 复核）

输入：Week2/data/unified/alpaca_all.json（download_data.py 产出）
输出：
  Week2/data/clean/alpaca_clean.json      清洗后 Alpaca 格式
  Week2/data/clean/sharegpt_clean.json    由清洗结果重建的 ShareGPT 格式
  Week2/deliverables/clean_funnel.png     清洗前后数量漏斗图
  Week2/deliverables/length_dist.png      清洗前后长度分布图
  Week2/deliverables/clean_stats.md/.json 统计报告

用法：
    .venv/Scripts/python.exe Week2/code/clean_pipeline.py
    # 用真实 Qwen 分词器计 token（更贴合训练 cutoff）：
    .venv/Scripts/python.exe Week2/code/clean_pipeline.py --tokenizer models/Qwen2.5-3B-Instruct
"""
import argparse
import hashlib
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UNI = ROOT / "Week2" / "data" / "unified" / "alpaca_all.json"
CLEAN_DIR = ROOT / "Week2" / "data" / "clean"
DELIV = ROOT / "Week2" / "deliverables"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
DELIV.mkdir(parents=True, exist_ok=True)

MAX_TOKENS = 2048          # 训练 cutoff_len，超过则截断
HAMMING_THRESH = 3         # SimHash 汉明距离 ≤ 该值判为近似重复
SIMHASH_BITS = 64
LSH_BANDS = 4              # 64bit 分 4 段，每段 16bit 做候选分桶

# ---------------- 文本清洗 ----------------
_TAG_RE = re.compile(r"<[^>]+>")                       # HTML 标签
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
# 不可见控制字符（保留 \n \t）+ 零宽字符 + BOM
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮﻿]")
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = _SCRIPT_RE.sub(" ", s)      # 先删 script/style 整块
    s = _TAG_RE.sub("", s)          # 再删普通标签
    s = html.unescape(s)            # &amp; &lt; -> & <
    s = _CTRL_RE.sub("", s)         # 删控制/零宽字符
    s = _WS_RE.sub(" ", s)          # 合并空格/tab
    s = _NL_RE.sub("\n\n", s)       # 3+ 连续换行压成 2
    return s.strip()


# ---------------- token 计数（三级回退）----------------
def build_token_counter(tokenizer_path):
    """返回 (count_fn, truncate_fn, backend_name)。
    优先真实 HF 分词器 → tiktoken → 中文友好启发式。"""
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            def count(t): return len(tok.encode(t))
            def trunc(t, n):
                ids = tok.encode(t)[:n]
                return tok.decode(ids, skip_special_tokens=True)
            return count, trunc, f"HF:{Path(tokenizer_path).name}"
        except Exception as e:
            print(f"[warn] 载入 HF 分词器失败({e})，回退 tiktoken")
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def count(t): return len(enc.encode(t))
        def trunc(t, n): return enc.decode(enc.encode(t)[:n])
        return count, trunc, "tiktoken:cl100k_base"
    except Exception as e:
        print(f"[warn] tiktoken 不可用({e})，回退字符启发式")
    # 启发式：CJK 每字≈1 token，其余按 4 字符≈1 token
    cjk = re.compile(r"[一-鿿　-〿＀-￯]")
    def count(t):
        n_cjk = len(cjk.findall(t))
        n_other = len(cjk.sub("", t))
        return n_cjk + n_other // 4 + 1
    def trunc(t, n):
        # 粗略按比例截断字符
        if count(t) <= n:
            return t
        ratio = n / max(count(t), 1)
        return t[: int(len(t) * ratio)]
    return count, trunc, "heuristic:cjk"


# ---------------- SimHash 去重 ----------------
def _shingles(text, k=3):
    """字符 k-gram（中文友好）。"""
    text = re.sub(r"\s+", "", text)
    if len(text) < k:
        return [text] if text else []
    return [text[i:i + k] for i in range(len(text) - k + 1)]


def simhash(text, bits=SIMHASH_BITS):
    v = [0] * bits
    grams = _shingles(text)
    if not grams:
        return 0
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= (1 << i)
    return out


def hamming(a, b):
    return bin(a ^ b).count("1")


def dedup_simhash(records, texts):
    """SimHash + LSH 分桶近似去重。返回保留索引列表 & 去重前后数。"""
    band_bits = SIMHASH_BITS // LSH_BANDS
    mask = (1 << band_bits) - 1
    hashes = [simhash(t) for t in texts]
    buckets = [dict() for _ in range(LSH_BANDS)]   # band -> {key: [idx,...]}
    kept, dropped = [], 0
    for idx, h in enumerate(hashes):
        is_dup = False
        cand = set()
        for b in range(LSH_BANDS):
            key = (h >> (b * band_bits)) & mask
            cand.update(buckets[b].get(key, ()))
        for j in cand:
            if hamming(h, hashes[j]) <= HAMMING_THRESH:
                is_dup = True
                break
        if is_dup:
            dropped += 1
            continue
        kept.append(idx)
        for b in range(LSH_BANDS):
            key = (h >> (b * band_bits)) & mask
            buckets[b].setdefault(key, []).append(idx)
    return kept, dropped


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="", help="HF 分词器路径（如 models/Qwen2.5-3B-Instruct）")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = ap.parse_args()

    if not UNI.exists():
        sys.exit(f"找不到输入 {UNI}，请先跑 download_data.py")

    data = json.loads(UNI.read_text(encoding="utf-8"))
    count_tok, trunc_tok, backend = build_token_counter(args.tokenizer)
    print(f"[tokenizer] {backend}")

    funnel = {"0_原始": len(data)}
    lengths_before = [count_tok((d.get("instruction", "") + d.get("input", "") + d.get("output", "")))
                      for d in data]

    # 1+2 清洗文本
    for d in data:
        d["instruction"] = clean_text(d.get("instruction", ""))
        d["input"] = clean_text(d.get("input", ""))
        d["output"] = clean_text(d.get("output", ""))

    # 3 空值过滤
    data = [d for d in data if d["instruction"] and d["output"]]
    funnel["1_去空值"] = len(data)

    # 4 超长截断
    trunc_cnt, drop_long = 0, 0
    kept = []
    for d in data:
        head = d["instruction"] + ("\n" + d["input"] if d["input"] else "")
        n_head = count_tok(head)
        if n_head >= args.max_tokens - 8:
            drop_long += 1
            continue                       # instruction 本身超长，丢
        budget = args.max_tokens - n_head - 8
        if count_tok(d["output"]) > budget:
            d["output"] = trunc_tok(d["output"], budget)
            trunc_cnt += 1
        kept.append(d)
    data = kept
    funnel["2_长度处理"] = len(data)

    # 5 SimHash 模糊去重
    texts = [d["instruction"] + " " + d["input"] + " " + d["output"] for d in data]
    keep_idx, n_dup = dedup_simhash(data, texts)
    data = [data[i] for i in keep_idx]
    funnel["3_去重后"] = len(data)

    lengths_after = [count_tok((d["instruction"] + d["input"] + d["output"])) for d in data]

    # 写清洗结果
    (CLEAN_DIR / "alpaca_clean.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    sharegpt = [{"conversations": [
        {"from": "human", "value": d["instruction"] + ("\n" + d["input"] if d["input"] else "")},
        {"from": "gpt", "value": d["output"]}]} for d in data]
    (CLEAN_DIR / "sharegpt_clean.json").write_text(
        json.dumps(sharegpt, ensure_ascii=False, indent=2), encoding="utf-8")

    # 统计 & 图
    stats = {
        "backend": backend, "max_tokens": args.max_tokens,
        "funnel": funnel, "truncated": trunc_cnt, "dropped_too_long": drop_long,
        "dropped_dup": n_dup,
        "len_before": {"min": min(lengths_before), "max": max(lengths_before),
                       "mean": round(sum(lengths_before) / len(lengths_before), 1)},
        "len_after": {"min": min(lengths_after), "max": max(lengths_after),
                      "mean": round(sum(lengths_after) / len(lengths_after), 1)},
    }
    (DELIV / "clean_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    write_stats_md(stats)
    make_plots(funnel, lengths_before, lengths_after, args.max_tokens)

    print("\n=== 清洗漏斗 ===")
    for k, v in funnel.items():
        print(f"  {k}: {v}")
    print(f"  截断: {trunc_cnt}  超长丢弃: {drop_long}  重复丢弃: {n_dup}")
    print(f"最终 {len(data)} 条 -> {(CLEAN_DIR / 'alpaca_clean.json').relative_to(ROOT)}")
    if len(data) >= 1500:
        print("✅ 达到验收②：清洗后 ≥1500 条")
    else:
        print(f"⚠️ 不足 1500 条（{len(data)}），可在 download 时增大取数")


def write_stats_md(s):
    f = s["funnel"]
    lines = ["# Day7 清洗统计报告\n",
             f"- 分词后端：`{s['backend']}`　最大 token：{s['max_tokens']}",
             "",
             "## 清洗漏斗（每步保留数）",
             "| 阶段 | 样本数 |", "|---|---|"]
    for k, v in f.items():
        lines.append(f"| {k} | {v} |")
    lines += ["",
              f"- 超长截断 output：**{s['truncated']}** 条",
              f"- instruction 超长丢弃：**{s['dropped_too_long']}** 条",
              f"- SimHash 模糊去重丢弃：**{s['dropped_dup']}** 条",
              "",
              "## 长度分布（token）",
              "| | min | mean | max |", "|---|---|---|---|",
              f"| 清洗前 | {s['len_before']['min']} | {s['len_before']['mean']} | {s['len_before']['max']} |",
              f"| 清洗后 | {s['len_after']['min']} | {s['len_after']['mean']} | {s['len_after']['max']} |",
              "", "## 图表", "- `clean_funnel.png` 清洗前后数量漏斗",
              "- `length_dist.png` 清洗前后长度分布直方图"]
    (DELIV / "clean_stats.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots(funnel, before, after, max_tokens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 漏斗
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = list(funnel.keys()); vs = list(funnel.values())
    bars = ax.bar(ks, vs, color="#4C78A8")
    ax.set_title("清洗漏斗：各阶段样本数")
    ax.set_ylabel("样本数")
    for b, v in zip(bars, vs):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(DELIV / "clean_funnel.png", dpi=130); plt.close(fig)

    # 长度分布
    fig, ax = plt.subplots(figsize=(8, 4))
    cap = max_tokens + 200
    bb = [min(x, cap) for x in before]; aa = [min(x, cap) for x in after]
    bins = 50
    ax.hist(bb, bins=bins, alpha=0.5, label=f"清洗前 (n={len(before)})", color="#E45756")
    ax.hist(aa, bins=bins, alpha=0.6, label=f"清洗后 (n={len(after)})", color="#4C78A8")
    ax.axvline(max_tokens, color="k", ls="--", lw=1, label=f"cutoff={max_tokens}")
    ax.set_title("清洗前后长度分布（token）")
    ax.set_xlabel("token 数"); ax.set_ylabel("样本数"); ax.legend()
    fig.tight_layout(); fig.savefig(DELIV / "length_dist.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
