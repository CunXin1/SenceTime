"""
build_calib_data.py — Week7 Day34
构造 PTQ 校准集（AWQ / GPTQ 共用），从 Week4 的偏好数据里抽。
Build the PTQ calibration set (shared by AWQ and GPTQ) from Week4 preference data.

★ 为什么校准集要用自己的领域数据，而不是 wikitext
    AWQ 和 GPTQ 都是"激活感知"的：AWQ 统计每个通道的激活幅度来决定哪些权重通道要放大
    保护（salient channel），GPTQ 用校准样本的 Hessian 逐列补偿量化误差。两者的统计量
    都来自**前向传播时真实的激活分布**——校准集分布偏离部署分布，保护错通道，量化误差就
    落在真正常用的通道上。wikitext 是英文百科散文，而这个模型部署后跑的是 Week3/Week4
    的中英混合对话，分布差得远。
    Both AWQ and GPTQ are activation-aware; their statistics come from real forward-pass
    activations. Calibrating on out-of-distribution text protects the wrong channels.

★ 为什么必须拼接成长文档（这是 LLaMA-Factory 的一个硬约束）
    LF 的 _get_quantization_dataset()（model/model_utils/quantization.py:60-66）是这么取样的：

        while True:
            sample = tokenizer(dataset[random_idx]["text"])
            if sample["input_ids"].size(1) > maxlen:
                break              # ← 只接受「比 maxlen 更长」的样本
            if n_try > 100:
                raise ValueError("Cannot find satisfying example, ...")

    它随机抽一条，**只有 token 数严格大于 export_quantization_maxlen 才接受**，然后从中
    随机截一个 maxlen 长的窗口。单条聊天样本通常只有几百 token，100 次重试全部落空，
    直接抛异常。所以这里把多条样本按 chat template 拼成一个长文档，保证每条都超过
    maxlen（留 TARGET_RATIO 的余量，避免 tokenizer 版本差异导致擦边失败）。
    LF only accepts samples LONGER than maxlen, so we concatenate turns into long docs.

★ 为什么排除 Week3 的 eval_questions.json
    那 20 道题是 Day35 要用来测生成质量的评测集。校准集看过评测题 = 数据泄漏，
    量化后的困惑度会虚低。校准和评测必须来自不相交的样本。
    The Week3 eval set is held out — calibrating on it would leak into the Day35 PPL.

用法 / Usage（Windows 主 .venv 即可，只依赖 transformers）:
    .venv/Scripts/python.exe Week7/code/build_calib_data.py
    .venv/Scripts/python.exe Week7/code/build_calib_data.py --maxlen 2048 --nsamples 256
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "Qwen2.5-3B-week4-dpo-merged"
PAIRS = ROOT / "Week4" / "data" / "dpo" / "dpo_pairs.json"
OUT = ROOT / "Week7" / "data" / "calib.json"
OUT_PPL = ROOT / "Week7" / "data" / "ppl_eval.json"

# 每个文档的目标长度 = maxlen * TARGET_RATIO。留 30% 余量：LF 的判定是严格大于，
# 擦边的文档一旦遇到 tokenizer 细微差异就会被拒，重试 100 次后整个导出失败。
TARGET_RATIO = 1.3
SEED = 42

# ★ 留出多少条原始样本给 PPL 评测（与校准池严格不相交）
#   Day35 要用困惑度衡量量化的质量损失。如果 PPL 评测文本被 AWQ/GPTQ 当校准集看过，
#   量化算法就是针对这批文本调的缩放/补偿，PPL 会虚低——量化看起来"几乎无损"，
#   其实只是过拟合了评测集。这里先把原始样本切成两个池子，再各自拼长文档，
#   保证两份产出没有任何一条共享的原始对话。
#   The PPL set must be disjoint from calibration, or AWQ/GPTQ tune on the very
#   text being scored and the quality loss looks artificially small.
HOLDOUT_RAW = 300


def render(pair: dict) -> str:
    """把一条偏好数据渲染成「用户提问 + 被选中的回答」的纯文本。

    只取 chosen 不取 rejected：校准要的是**部署后模型实际会产生的**激活分布，
    而 DPO 之后模型输出的就是 chosen 那一侧的风格。喂 rejected 等于让量化去
    保护模型已经被训得不再走的那条路径。
    Only `chosen` is used — it matches the post-DPO output distribution.
    """
    convs = pair.get("conversations") or []
    user = next((c["value"] for c in convs if c.get("from") == "human"), "")
    answer = (pair.get("chosen") or {}).get("value", "")
    if not user or not answer:
        return ""
    # 用 Qwen 的 ChatML 标记手工拼，而不是 apply_chat_template：这里要的是拼接成
    # 长文档，逐条套模板会在文档中间插入大量 system 段，稀释真实对话的激活统计。
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n"


def pack(texts: list[str], tok, n_docs: int, target_tokens: int, seed: int) -> list[dict[str, str]]:
    """把短样本按顺序拼成 n_docs 个「长度超过 target_tokens」的长文档。"""
    docs: list[dict[str, str]] = []
    buf: list[str] = []
    buf_tokens = 0
    cursor = 0
    while len(docs) < n_docs:
        if cursor >= len(texts):          # 样本用完就再洗一轮继续拼
            random.Random(seed + len(docs)).shuffle(texts)
            cursor = 0
        piece = texts[cursor]
        cursor += 1
        buf.append(piece)
        buf_tokens += len(tok(piece, add_special_tokens=False)["input_ids"])
        if buf_tokens > target_tokens:
            docs.append({"text": "".join(buf)})
            buf, buf_tokens = [], 0
    return docs


def check(docs: list[dict[str, str]], tok, maxlen: int, label: str) -> list[int]:
    """复核长度，顺便返回长度分布。

    GPTQ 那条路径在取样失败时会等到模型加载完、跑到一半才抛异常，白等十几分钟。
    这里提前一次性验完。
    """
    lens = [len(tok(d["text"], add_special_tokens=False)["input_ids"]) for d in docs]
    too_short = [n for n in lens if n <= maxlen]
    if too_short:
        raise SystemExit(f"[FAIL] {label}: {len(too_short)} 条文档 <= maxlen({maxlen})，"
                         f"最短 {min(lens)}。调大 TARGET_RATIO 后重跑。")
    return lens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxlen", type=int, default=1024,
                    help="必须与 export_gptq_w4.yaml 的 export_quantization_maxlen 一致")
    ap.add_argument("--nsamples", type=int, default=256,
                    help="校准集产出多少个长文档；GPTQ 实际只随机抽 nsamples 条，多备无妨")
    ap.add_argument("--n-ppl", type=int, default=48,
                    help="PPL 评测集产出多少个长文档")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--out-ppl", type=Path, default=OUT_PPL)
    args = ap.parse_args()

    target_tokens = int(args.maxlen * TARGET_RATIO)
    print(f"[cfg] maxlen={args.maxlen} 目标文档长度>{target_tokens} tok")

    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    print(f"[src] {PAIRS.relative_to(ROOT)}: {len(pairs)} 条偏好对")

    texts = [t for t in (render(p) for p in pairs) if t]
    random.Random(SEED).shuffle(texts)  # 固定种子，两份产出都可复现

    # 先切池子再拼文档：保证 calib 和 ppl_eval 不共享任何一条原始对话
    ppl_pool, calib_pool = texts[:HOLDOUT_RAW], texts[HOLDOUT_RAW:]
    print(f"[split] 校准池 {len(calib_pool)} 条 / PPL 池 {len(ppl_pool)} 条（互斥）")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for pool, n_docs, out_path, label in (
        (calib_pool, args.nsamples, args.out, "calib"),
        (ppl_pool, args.n_ppl, args.out_ppl, "ppl_eval"),
    ):
        docs = pack(pool, tok, n_docs, target_tokens, SEED)
        lens = check(docs, tok, args.maxlen, label)
        out_path.write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[ok] {out_path.relative_to(ROOT)}: {len(docs)} 条, "
              f"token min={min(lens)} / 中位={sorted(lens)[len(lens) // 2]} / max={max(lens)}")
    print(f"[ok] 两份产出长度全部 > maxlen({args.maxlen})，LF 取样不会拒绝")


if __name__ == "__main__":
    main()
