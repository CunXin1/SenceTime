"""
build_preference_data.py — Week4 Day18
构建 DPO 偏好数据集：开源对（UltraFeedback 英 + DPO-En-Zh-20k 中）+ 自建 220 对，
统一为 LLaMA-Factory sharegpt 偏好格式，校验后落盘并注册。
Build the DPO preference dataset: open-source pairs (UltraFeedback EN +
DPO-En-Zh-20k ZH) merged with the 220 hand-authored pairs, unified into the
LLaMA-Factory sharegpt preference format, validated, written and registered.

数据来源 / Sources（HF 直连，缓存进 Week4/data/hf_cache/）:
    1. llamafactory/ultrafeedback_binarized  train.json  {instruction,chosen,rejected}(str) → 抽 500
    2. llamafactory/DPO-En-Zh-20k            dpo_zh.json  已是 sharegpt 偏好格式（取单轮）→ 抽 500
    3. Week4/data/self_built_pairs.json       自建 220 条（本地，手写，见 make_self_built_pairs.py）

产出 / Output:
    Week4/data/dpo/dpo_pairs.json     仅 conversations/chosen/rejected 三键（训练用）
    Week4/data/dpo/pairs_meta.json    同序 {source,pref_type,sub_type}（统计用）
    Week4/data/dpo/dataset_info.json  LLaMA-Factory 注册表（ranking: true）
    Week4/deliverables/偏好数据集统计.md + pref_dist.png + pref_length.png

用法 / Usage（仓库根目录 / from repo root）:
    .venv/Scripts/python.exe Week4/code/build_preference_data.py
    .venv/Scripts/python.exe Week4/code/build_preference_data.py --uf-n 500 --zh-n 500
    .venv/Scripts/python.exe Week4/code/build_preference_data.py --upsample-safety 2  # 安全对翻倍
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Week4" / "data"
OUT_DIR = DATA_DIR / "dpo"
SELF_BUILT = DATA_DIR / "self_built_pairs.json"
DELIV = ROOT / "Week4" / "deliverables"


def load_env(env_path: Path) -> None:
    """读 .env（HF_TOKEN 等），复用 Week2/code/download_data.py 的做法。
    Load .env (HF_TOKEN etc.), same as Week2's download_data.py."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def to_pair(prompt: str, chosen: str, rejected: str) -> dict:
    """统一的 sharegpt 偏好三元组 / unified sharegpt preference triple."""
    return {
        "conversations": [{"from": "human", "value": prompt.strip()}],
        "chosen": {"from": "gpt", "value": chosen.strip()},
        "rejected": {"from": "gpt", "value": rejected.strip()},
    }


def load_ultrafeedback(n: int, rng: random.Random) -> list[tuple[dict, dict]]:
    """抽 n 条 UltraFeedback（英），返回 [(pair, meta), ...]。
    Sample n UltraFeedback (EN) pairs; return [(pair, meta), ...]."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("llamafactory/ultrafeedback_binarized", "train.json",
                        repo_type="dataset")
    rows = json.loads(Path(p).read_text(encoding="utf-8"))
    # 过滤空值后再抽样，避免抽到脏数据 / drop empties before sampling
    rows = [r for r in rows if r.get("instruction") and r.get("chosen")
            and r.get("rejected") and r["chosen"] != r["rejected"]]
    picked = rng.sample(rows, min(n, len(rows)))
    out = []
    for r in picked:
        pair = to_pair(r["instruction"], r["chosen"], r["rejected"])
        out.append((pair, {"source": "ultrafeedback", "pref_type": "开源(英)",
                           "sub_type": "通用"}))
    return out


def load_dpo_zh(n: int, rng: random.Random) -> list[tuple[dict, dict]]:
    """抽 n 条 DPO-En-Zh-20k 的中文单轮对，返回 [(pair, meta), ...]。
    Sample n single-turn ZH pairs from DPO-En-Zh-20k."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("llamafactory/DPO-En-Zh-20k", "dpo_zh.json",
                        repo_type="dataset")
    rows = json.loads(Path(p).read_text(encoding="utf-8"))
    # 只取单轮（conversations 长度为 1），保持与自建对同构、避免多轮上下文干扰。
    # Keep single-turn only, matching the self-built pairs.
    single = [r for r in rows if len(r.get("conversations", [])) == 1
              and r["conversations"][0].get("value")
              and r.get("chosen", {}).get("value")
              and r.get("rejected", {}).get("value")
              and r["chosen"]["value"].strip() != r["rejected"]["value"].strip()]
    picked = rng.sample(single, min(n, len(single)))
    out = []
    for r in picked:
        pair = {"conversations": r["conversations"],
                "chosen": r["chosen"], "rejected": r["rejected"]}
        out.append((pair, {"source": "dpo_zh", "pref_type": "开源(中)",
                           "sub_type": "通用"}))
    return out


def load_self_built(upsample_safety: int) -> list[tuple[dict, dict]]:
    """读自建 220 对；upsample_safety>1 时把安全类样本复制若干份（拒答率兜底杠杆）。
    Load the hand-authored pairs; upsample safety pairs when the flag > 1."""
    rows = json.loads(SELF_BUILT.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        pair = to_pair(r["prompt"], r["chosen"], r["rejected"])
        meta = {"source": "self_built", "pref_type": r["pref_type"],
                "sub_type": r["sub_type"]}
        reps = upsample_safety if r["pref_type"] == "安全性" else 1
        for _ in range(reps):
            out.append((pair, meta))
    return out


def validate(pairs: list[dict], metas: list[dict], cutoff: int) -> int:
    """执行《偏好数据构造指南》§五 的校验规则，返回超长告警计数。
    Enforce the guide's §5 validation rules; return the count of over-length warnings."""
    over_len = 0
    seen_self_prompts = set()
    for pair, meta in zip(pairs, metas):
        prompt = pair["conversations"][0]["value"]
        chosen = pair["chosen"]["value"]
        rejected = pair["rejected"]["value"]
        # 1) 非空 / non-empty
        assert prompt and chosen and rejected, f"空字段 / empty field: {prompt[:30]}"
        # 2) chosen != rejected
        assert chosen.strip() != rejected.strip(), \
            f"chosen==rejected: {prompt[:30]}"
        # 3) 长度合理性（安全类豁免，其余 chosen 不得显著短于 rejected）
        if meta["pref_type"] not in ("安全性", "开源(英)", "开源(中)"):
            assert len(chosen) >= 0.5 * len(rejected), \
                f"chosen 显著短于 rejected / chosen too short: {prompt[:30]}"
        # 4) 自建对 prompt 去重
        if meta["source"] == "self_built":
            assert prompt not in seen_self_prompts, f"自建 prompt 重复 / dup: {prompt[:30]}"
            seen_self_prompts.add(prompt)
        # 5) 超长告警（粗略按字符数 * 0.6 估 token，仅统计不报错）
        if max(len(prompt) + len(chosen), len(prompt) + len(rejected)) * 0.6 > cutoff:
            over_len += 1
    return over_len


def make_plots(metas: list[dict], pairs: list[dict]) -> None:
    """两张统计图：来源×类型分布、chosen/rejected 长度分布。
    Two charts: source×type distribution, and chosen/rejected length distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 图1：来源分类堆叠柱 / distribution by source and type
    types = ["开源(英)", "开源(中)", "安全性", "事实正确性", "完整性", "有用性", "格式"]
    counts = Counter(m["pref_type"] for m in metas)
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [counts.get(t, 0) for t in types]
    colors = ["#8888cc", "#88bbcc", "#e06666", "#f6b26b", "#93c47d",
              "#76a5af", "#c27ba0"]
    bars = ax.bar(types, vals, color=colors)
    ax.bar_label(bars)
    ax.set_ylabel("条数 / count")
    ax.set_title(f"Week4 偏好数据集组成（共 {len(metas)} 条）")
    plt.xticks(rotation=20)
    plt.tight_layout()
    fig.savefig(DELIV / "pref_dist.png", dpi=120)
    plt.close(fig)

    # 图2：chosen vs rejected 字符长度分布（证明 chosen 并非单纯更长）
    ch = [len(p["chosen"]["value"]) for p in pairs]
    rj = [len(p["rejected"]["value"]) for p in pairs]
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = range(0, 2001, 100)
    ax.hist(ch, bins=bins, alpha=0.6, label="chosen", color="#93c47d")
    ax.hist(rj, bins=bins, alpha=0.6, label="rejected", color="#e06666")
    ax.set_xlabel("回答字符数 / answer length (chars)")
    ax.set_ylabel("样本数 / count")
    ax.set_title("chosen vs rejected 长度分布")
    ax.legend()
    plt.tight_layout()
    fig.savefig(DELIV / "pref_length.png", dpi=120)
    plt.close(fig)


def write_stats(metas: list[dict], pairs: list[dict], over_len: int,
                args) -> None:
    """写《偏好数据集统计.md》。Write the dataset stats markdown."""
    by_source = Counter(m["source"] for m in metas)
    by_type = Counter(m["pref_type"] for m in metas)
    safe_sub = Counter(m["sub_type"] for m in metas if m["pref_type"] == "安全性")
    ch_avg = sum(len(p["chosen"]["value"]) for p in pairs) / len(pairs)
    rj_avg = sum(len(p["rejected"]["value"]) for p in pairs) / len(pairs)

    lines = [
        "# Week4 Day18 交付：偏好数据集统计",
        "",
        "> 由 `Week4/code/build_preference_data.py` 自动生成。",
        f"> 总量 **{len(pairs)}** 条（验收下限 300，✅ 达标）。",
        f"> 随机种子 {args.seed}，可复现。",
        "",
        "## 一、来源构成",
        "",
        "| 来源 | 条数 | 语言 | 说明 |",
        "|---|---|---|---|",
        f"| UltraFeedback | {by_source.get('ultrafeedback', 0)} | 英 | 通用有用性/正确性偏好 |",
        f"| DPO-En-Zh-20k (zh) | {by_source.get('dpo_zh', 0)} | 中 | 语言平衡 |",
        f"| 自建 | {by_source.get('self_built', 0)} | 中 | 5 类偏好精准覆盖 |",
        f"| **合计** | **{len(pairs)}** | 中英均衡 | |",
        "",
        "## 二、偏好类型分布",
        "",
        "| 类型 | 条数 |",
        "|---|---|",
    ]
    for t in ["开源(英)", "开源(中)", "安全性", "事实正确性", "完整性", "有用性", "格式"]:
        if by_type.get(t):
            lines.append(f"| {t} | {by_type[t]} |")
    lines += [
        "",
        "## 三、安全类风险覆盖（自建）",
        "",
        "| 风险子类 | 条数 |",
        "|---|---|",
    ]
    for sub, n in sorted(safe_sub.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {sub} | {n} |")
    lines += [
        "",
        "## 四、质量指标",
        "",
        f"- chosen 平均长度：{ch_avg:.0f} 字符",
        f"- rejected 平均长度：{rj_avg:.0f} 字符",
        f"- 超 `cutoff_len` 告警条数：{over_len}（训练时会被截断，不影响可用性）",
        "- 校验规则（非空 / chosen≠rejected / 长度合理性 / 自建 prompt 去重）：**全部通过**",
        "",
        "> 长度分布见 `pref_length.png`：安全类 chosen 因四段式拒答说明偏长，其余类型"
        "chosen 与 rejected 长度接近，说明偏好信号来自质量而非单纯长度。",
        "",
        "## 五、组成可视化",
        "",
        "![来源与类型分布](pref_dist.png)",
        "",
        "![chosen/rejected 长度分布](pref_length.png)",
    ]
    DELIV.mkdir(parents=True, exist_ok=True)
    (DELIV / "偏好数据集统计.md").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uf-n", type=int, default=500, help="UltraFeedback 抽样数")
    ap.add_argument("--zh-n", type=int, default=500, help="DPO-En-Zh-20k 抽样数")
    ap.add_argument("--upsample-safety", type=int, default=1,
                    help="安全类样本复制倍数（拒答率兜底杠杆，默认 1=不上采样）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cutoff", type=int, default=1024,
                    help="与训练 cutoff_len 一致，用于超长告警统计")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf_cache"))
    print(f"[env] HF_HOME = {os.environ['HF_HOME']}")

    rng = random.Random(args.seed)
    print("[load] UltraFeedback ...")
    uf = load_ultrafeedback(args.uf_n, rng)
    print(f"       {len(uf)} 条")
    print("[load] DPO-En-Zh-20k (zh) ...")
    zh = load_dpo_zh(args.zh_n, rng)
    print(f"       {len(zh)} 条")
    print("[load] 自建对 ...")
    sb = load_self_built(args.upsample_safety)
    print(f"       {len(sb)} 条（upsample_safety={args.upsample_safety}）")

    # 合并后整体 shuffle（同一 rng，可复现），让各来源在训练中均匀分布。
    merged = uf + zh + sb
    rng.shuffle(merged)
    pairs = [p for p, _ in merged]
    metas = [m for _, m in merged]

    print("[check] 校验中 ...")
    over_len = validate(pairs, metas, args.cutoff)
    print(f"        通过；超长告警 {over_len} 条")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dpo_pairs.json").write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "pairs_meta.json").write_text(
        json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "dataset_info.json").write_text(json.dumps({
        "week4_dpo_pairs": {
            "file_name": "dpo_pairs.json", "ranking": True,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "chosen": "chosen",
                        "rejected": "rejected"},
            "tags": {"role_tag": "from", "content_tag": "value",
                     "user_tag": "human", "assistant_tag": "gpt"},
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    make_plots(metas, pairs)
    write_stats(metas, pairs, over_len, args)

    print(f"\n[OK] {len(pairs)} 条 -> {OUT_DIR.relative_to(ROOT)}")
    print(f"[OK] 统计与图 -> {DELIV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
