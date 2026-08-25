"""step1_data_prep.py — Week8 Day40 / 任务书 40.1

一条命令把「原始数据 → 清洗 → 去重 → 双格式转换 → 9:1 划分 → 注册给
LLaMA-Factory → 统计报告」全部跑完。step2_train.sh 只需要 dataset_dir
指向本脚本的 --out-dir，不再需要任何人工干预。
One command: raw data -> clean -> dedup -> dual-format -> 9:1 split ->
LLaMA-Factory registration -> stats report.

--------------------------------------------------------------------------
★ 取舍一：清洗逻辑**复用** Week2/code/clean_pipeline.py，不复制粘贴
    第 2 周那套清洗（HTML 剥离 / 控制字符 / 空值 / 超长截断 / SimHash+LSH 模糊
    去重）已经被 Week3、Week4 的全部实验验证过，产出的 4684 条是那两周所有
    结论的数据基础。如果这里再抄一份代码，两份实现迟早漂移，Week8 的自动化流水线
    就会训出一个"和第 3 周最优超参对应的数据"不一样的模型，超参的最优性也就
    不成立了。做法是把 Week2/code 塞进 sys.path 直接 import 它的函数
    （clean_text / build_token_counter / dedup_simhash / simhash）。
    clean_pipeline.py 顶层只做 mkdir(exist_ok=True)、把 main() 放在
    `if __name__ == "__main__"` 里，import 它没有副作用——这一点已确认。
    Reuse, never copy: a forked cleaner would silently drift from the data
    that Week3/Week4's "best hyper-parameters" were tuned on.

★ 取舍二：显式 9:1 划分 vs LLaMA-Factory 的 val_size —— 二选一，不能并存
    Week3 最优配置写的是 `val_size: 0.05`，那是 LF **在加载数据之后自己**再切
    一刀。两套机制放在一起有两种失败形态，都很难受：
      · 同时写 `val_size` 和 `eval_dataset` —— LF 直接抛异常，训练根本起不来。
        （LLaMA-Factory/src/llamafactory/hparams/data_args.py:162
         `if self.eval_dataset is not None and self.val_size > 1e-6: raise
          ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")`）
      · 只留 `val_size`、不写 `eval_dataset` —— 这个更阴险，它**不报错**：
        本脚本切出来的 val.json 一条都不会被用到，LF 又在 train 上偷偷切 5%，
        实际参与训练的样本比预期再少 4.5%，与第 3 周的最优实验不再等价；
        而报告里那份漂亮的验证集统计描述的是一个从未被评测过的集合。
        LF 那一刀的随机性还由它自己的 seed 控制，复现口径又多一处分叉。
    ✅ 本周方案：**在 step1 里显式划分，并在 sft_best.yaml / dpo_best.yaml 里
       删掉 val_size，改用 `eval_dataset: week8_sft_val`**。
       理由是划分要可审计——val.json 是落盘的实体文件，能被 step3 直接拿去评测，
       也能在报告里给出它自己的长度/来源分布；LF 内部切分是个黑盒，出了问题
       没法复查。代价是验证集比例从 5% 变成 10%（任务书 40.1 指定 9:1），
       训练样本少了约 5%，属于任务书规定的口径变化，已在文档中写明。
    ✅ Explicit split here + `eval_dataset:` in the YAML; `val_size` REMOVED.
       Keeping both would double-split and silently shrink the training set.

★ 取舍三：来源分布靠 manifest 的顺序区间还原，而不是给每条打标签
    Week2 的 alpaca_all.json 只有 {instruction,input,output} 三个字段，没有
    source 标签。但 download_data.py 是**按 DATASETS 顺序依次 append** 的，
    manifest.json 里每个来源的 valid 就是它连续占用的条数。所以
    [0,2000) = alpaca_gpt4_zh，[2000,3976) = coig_pc，[3976,4975) = sharegpt_zh。
    脚本会先校验 sum(valid) == len(data)，对不上就整体标 "unknown" 而不是
    硬套区间——宁可少报一个维度，也不能报一个错的维度。
    Source labels are reconstructed from manifest index ranges, guarded by a
    sum check; falls back to "unknown" rather than guessing.

★ 取舍四：DPO 偏好数据也在这里划分
    任务书 40.2 要求 step2 依次跑 SFT 和 DPO，而两个 stage 的配置都必须吃
    Week8/data。偏好对的 schema（conversations/chosen/rejected）和 Alpaca 不同，
    不能走同一条清洗流水线，但**文本清洗和 9:1 划分是共用的**：这里对
    conversations/chosen/rejected 的每个 value 跑同一个 clean_text，按 prompt
    做 SimHash 去重，再用同一个 seed 切 9:1。这样 SFT 和 DPO 的验证集划分口径
    完全一致，step3 的两条评测曲线才可比。

用法 / Usage:
    .venv/Scripts/python.exe Week8/scripts/step1_data_prep.py
    # 冒烟（只取前 200 条，秒级跑完，用来验证链路）:
    .venv/Scripts/python.exe Week8/scripts/step1_data_prep.py --quick 200
    # 换分词器 / 换比例:
    .venv/Scripts/python.exe Week8/scripts/step1_data_prep.py --val-ratio 0.1 --seed 42

产物 / Outputs（全部在 --out-dir，默认 Week8/data）:
    train.json / val.json                   Alpaca 格式，9:1
    train_sharegpt.json / val_sharegpt.json ShareGPT 格式（同一批样本的另一种编码）
    dpo_train.json / dpo_val.json           偏好对，9:1
    dataset_info.json                       LLaMA-Factory 注册表（6 个数据集全注册）
    ../deliverables/data_stats.json / .md   统计报告
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

SCRIPT_VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parents[2]

# ★ 把 Week2/code 挂到 sys.path 前面，直接 import 第 2 周的清洗实现。
#   用 sys.path 而不是 importlib.util.spec_from_file_location，是因为
#   clean_pipeline.py 自己也要 import transformers/tiktoken，走正常的模块机制
#   最省事，也让 traceback 里显示它的真实文件名，好排查。
sys.path.insert(0, str(ROOT / "Week2" / "code"))
try:
    import clean_pipeline as cp   # noqa: E402  (必须在改 sys.path 之后)
except Exception as exc:          # pragma: no cover
    sys.exit(
        f"[FATAL] 无法 import Week2/code/clean_pipeline.py：{exc}\n"
        f"        本脚本刻意不复制它的清洗逻辑（见文件头 ★取舍一）。\n"
        f"        请确认 {ROOT / 'Week2' / 'code' / 'clean_pipeline.py'} 存在。"
    )


# ---------------------------------------------------------------------------
# 工具 / helpers
# ---------------------------------------------------------------------------
def pct(values, q):
    """分位数。用 statistics.quantiles 而不是 numpy —— 这个脚本定位是「零重依赖」，
    不该为了一个 p90 就把 numpy 拉进 import 链（.venv 里虽然有，但 step1 应该
    在任何一个干净的 python 里都能跑，方便在别的机器上重放数据准备）。"""
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    # n=100 把数据切成 100 份、给出 99 个分界点，取第 q 个即第 q 百分位
    return round(statistics.quantiles(values, n=100, method="inclusive")[q - 1], 1)


def dist(values):
    """min / median / p90 / max / mean。长度分布统一走这一个结构，
    JSON 和 Markdown 两处渲染共用，避免两边字段名对不上。"""
    if not values:
        return {"n": 0, "min": 0, "median": 0, "p90": 0, "max": 0, "mean": 0}
    return {
        "n": len(values),
        "min": min(values),
        "median": round(statistics.median(values), 1),
        "p90": pct(values, 90),
        "max": max(values),
        "mean": round(sum(values) / len(values), 1),
    }


def load_source_labels(n_records):
    """用 Week2 的 manifest.json 还原每条样本的来源（见文件头 ★取舍三）。"""
    mf = ROOT / "Week2" / "data" / "unified" / "manifest.json"
    if not mf.exists():
        return ["unknown"] * n_records
    try:
        entries = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return ["unknown"] * n_records
    ok = [e for e in entries if e.get("status") == "OK" and "valid" in e]
    if sum(e["valid"] for e in ok) != n_records:
        # 条数对不上说明 manifest 与 alpaca_all.json 不是同一次产出的，
        # 区间映射不再可信 —— 宁可不报这个维度，也不能报一个错的。
        print(f"[warn] manifest 的 valid 之和 != 样本数({n_records})，来源分布标为 unknown")
        return ["unknown"] * n_records
    labels = []
    for e in ok:
        labels.extend([e["name"]] * e["valid"])
    return labels


def to_sharegpt(rec):
    """Alpaca -> ShareGPT。instruction 与 input 用换行拼成一条 human 消息 ——
    与 Week2 clean_pipeline.py 的转换方式保持完全一致，两周的 ShareGPT 文件
    才是同一个口径（否则拿两边的数据对比会发现 prompt 长度莫名其妙对不上）。"""
    human = rec["instruction"] + ("\n" + rec["input"] if rec.get("input") else "")
    return {"conversations": [
        {"from": "human", "value": human},
        {"from": "gpt", "value": rec["output"]},
    ]}


def split_9_1(items, val_ratio, seed):
    """随机划分。★ 用 random.Random(seed) 造独立实例而不是 random.seed()：
    全局种子会被后续任何一次 random 调用（包括第三方库内部的）污染，独立实例
    保证同一个 seed 永远给出同一个划分，这是「可复现」的最低要求。"""
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, round(len(items) * val_ratio)) if items else 0
    train = [items[i] for i in idx[n_val:]]
    val = [items[i] for i in idx[:n_val]]
    return train, val


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# SFT 数据：清洗漏斗
# ---------------------------------------------------------------------------
def prepare_sft(args, count_tok, trunc_tok, timings):
    src = Path(args.source)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        sys.exit(
            f"[FATAL] 找不到原始数据 {src}\n"
            f"        它由 Week2/code/download_data.py 产出。请先跑：\n"
            f"          .venv/Scripts/python.exe Week2/code/download_data.py\n"
            f"        （需要能访问 HuggingFace；离线环境下把已有的 alpaca_all.json\n"
            f"          拷到该路径，或用 --source 指向别处的同格式文件）"
        )

    t0 = time.time()
    data = json.loads(src.read_text(encoding="utf-8"))
    labels = load_source_labels(len(data))
    if args.quick:
        # 冒烟：只取前 N 条。来源标签必须同步截断，否则统计会整体错位。
        data, labels = data[: args.quick], labels[: args.quick]
    timings["load"] = round(time.time() - t0, 2)

    funnel = {"0_raw": len(data)}
    src_raw = Counter(labels)

    # --- 步骤 1+2：HTML 剥离 + 控制字符 + 空白规范（复用 cp.clean_text）---
    t0 = time.time()
    changed = 0
    for d in data:
        before = (d.get("instruction", ""), d.get("input", ""), d.get("output", ""))
        d["instruction"] = cp.clean_text(d.get("instruction", ""))
        d["input"] = cp.clean_text(d.get("input", ""))
        d["output"] = cp.clean_text(d.get("output", ""))
        if (d["instruction"], d["input"], d["output"]) != before:
            changed += 1
    funnel["1_text_cleaned"] = len(data)          # 该步只改文本，不丢样本
    timings["clean_text"] = round(time.time() - t0, 2)

    # --- 步骤 3：空值过滤 ---
    keep = [i for i, d in enumerate(data) if d["instruction"] and d["output"]]
    data = [data[i] for i in keep]
    labels = [labels[i] for i in keep]
    funnel["2_drop_empty"] = len(data)

    # --- 步骤 4：超长处理（instruction 本身超长则丢，否则截 output 尾部）---
    t0 = time.time()
    trunc_cnt, drop_long = 0, 0
    keep_data, keep_lab = [], []
    for d, lb in zip(data, labels):
        head = d["instruction"] + ("\n" + d["input"] if d["input"] else "")
        n_head = count_tok(head)
        if n_head >= args.max_tokens - 8:
            drop_long += 1
            continue
        budget = args.max_tokens - n_head - 8      # 留 8 token 给 ChatML 模板的特殊符号
        if count_tok(d["output"]) > budget:
            d["output"] = trunc_tok(d["output"], budget)
            trunc_cnt += 1
        keep_data.append(d)
        keep_lab.append(lb)
    data, labels = keep_data, keep_lab
    funnel["3_length_handled"] = len(data)
    timings["length"] = round(time.time() - t0, 2)

    # --- 步骤 5：SimHash + LSH 模糊去重（复用 cp.dedup_simhash）---
    t0 = time.time()
    texts = [d["instruction"] + " " + d.get("input", "") + " " + d["output"] for d in data]
    keep_idx, n_dup = cp.dedup_simhash(data, texts)
    data = [data[i] for i in keep_idx]
    labels = [labels[i] for i in keep_idx]
    funnel["4_deduped"] = len(data)
    timings["dedup"] = round(time.time() - t0, 2)

    # --- 步骤 6：9:1 随机划分 ---
    t0 = time.time()
    paired = list(zip(data, labels))
    tr_pair, va_pair = split_9_1(paired, args.val_ratio, args.seed)
    train = [p[0] for p in tr_pair]
    val = [p[0] for p in va_pair]
    tr_lab = [p[1] for p in tr_pair]
    va_lab = [p[1] for p in va_pair]
    funnel["5_train"] = len(train)
    funnel["5_val"] = len(val)
    timings["split"] = round(time.time() - t0, 2)

    stats = {
        "modified_by_clean_text": changed,
        "truncated": trunc_cnt,
        "dropped_too_long": drop_long,
        "dropped_dup": n_dup,
        "source_raw": dict(src_raw.most_common()),
        "source_train": dict(Counter(tr_lab).most_common()),
        "source_val": dict(Counter(va_lab).most_common()),
    }
    return train, val, funnel, stats


# ---------------------------------------------------------------------------
# DPO 偏好数据：同一套文本清洗 + 同一个 seed 的 9:1
# ---------------------------------------------------------------------------
def prepare_dpo(args, count_tok, timings):
    src = Path(args.dpo_source)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        print(f"[warn] 找不到偏好数据 {src}，跳过 DPO 划分（step2 的 dpo 阶段将无数据可用）")
        return None, None, None

    t0 = time.time()
    pairs = json.loads(src.read_text(encoding="utf-8"))
    if args.quick:
        pairs = pairs[: args.quick]
    funnel = {"0_raw": len(pairs)}

    cleaned = []
    for p in pairs:
        convs = [{"from": m.get("from", "human"), "value": cp.clean_text(m.get("value", ""))}
                 for m in p.get("conversations", [])]
        chosen = dict(p.get("chosen", {}))
        rejected = dict(p.get("rejected", {}))
        chosen["value"] = cp.clean_text(chosen.get("value", ""))
        rejected["value"] = cp.clean_text(rejected.get("value", ""))
        if not convs or not any(c["value"] for c in convs):
            continue
        # chosen / rejected 任一为空这条对就废了：DPO 的 loss 要用两者的
        # log-ratio 之差，缺一边算不出来。
        if not chosen["value"] or not rejected["value"]:
            continue
        # chosen == rejected 提供不了任何偏好信号，梯度恒为 0，纯粹浪费算力。
        if chosen["value"] == rejected["value"]:
            continue
        cleaned.append({"conversations": convs, "chosen": chosen, "rejected": rejected})
    funnel["1_cleaned"] = len(cleaned)

    # ★ 只按 prompt 去重（不含 chosen/rejected）：同一个 prompt 配不同回答，
    #   在偏好数据里等于对该 prompt 重复采样，会让它在 loss 里被加权多次，
    #   偏好信号被少数高频 prompt 主导。
    prompts = [" ".join(c["value"] for c in p["conversations"]) for p in cleaned]
    keep_idx, n_dup = cp.dedup_simhash(cleaned, prompts)
    cleaned = [cleaned[i] for i in keep_idx]
    funnel["2_deduped"] = len(cleaned)

    train, val = split_9_1(cleaned, args.val_ratio, args.seed)
    funnel["3_train"] = len(train)
    funnel["3_val"] = len(val)
    timings["dpo"] = round(time.time() - t0, 2)

    stats = {"funnel": funnel, "dropped_dup": n_dup}
    for name, recs in (("train", train), ("val", val)):
        ls = [count_tok(" ".join(c["value"] for c in r["conversations"]) + r["chosen"]["value"])
              for r in recs]
        stats[f"len_{name}_token"] = dist(ls)
    return train, val, stats


# ---------------------------------------------------------------------------
# LLaMA-Factory 注册表
# ---------------------------------------------------------------------------
def build_dataset_info():
    """写 dataset_info.json。train / val 都注册，让 sft_best.yaml 能直接写
    `dataset: week8_sft_train` + `eval_dataset: week8_sft_val`（见 ★取舍二）。
    字段名与 Week2/Week4 的注册表保持一致，step3 可以复用同一套读取代码。"""
    alpaca_cols = {"prompt": "instruction", "query": "input", "response": "output"}
    sharegpt_tags = {"role_tag": "from", "content_tag": "value",
                     "user_tag": "human", "assistant_tag": "gpt"}
    sharegpt_cols = {"messages": "conversations"}
    return {
        "week8_sft_train": {"file_name": "train.json", "columns": alpaca_cols},
        "week8_sft_val": {"file_name": "val.json", "columns": alpaca_cols},
        # ShareGPT 版本是同一批样本的另一种编码（任务书 40.1 的"格式转换"）。
        # 本周 sft_best.yaml 走 Alpaca 版，与 Week3 最优实验口径一致。
        "week8_sft_train_sharegpt": {
            "file_name": "train_sharegpt.json", "formatting": "sharegpt",
            "columns": sharegpt_cols, "tags": sharegpt_tags},
        "week8_sft_val_sharegpt": {
            "file_name": "val_sharegpt.json", "formatting": "sharegpt",
            "columns": sharegpt_cols, "tags": sharegpt_tags},
        # ranking: true 是 DPO 的必需项，缺了 LF 会按普通 SFT 数据加载然后报错。
        "week8_dpo_train": {
            "file_name": "dpo_train.json", "ranking": True, "formatting": "sharegpt",
            "columns": {"messages": "conversations", "chosen": "chosen",
                        "rejected": "rejected"}, "tags": sharegpt_tags},
        "week8_dpo_val": {
            "file_name": "dpo_val.json", "ranking": True, "formatting": "sharegpt",
            "columns": {"messages": "conversations", "chosen": "chosen",
                        "rejected": "rejected"}, "tags": sharegpt_tags},
    }


# ---------------------------------------------------------------------------
# 人读报告
# ---------------------------------------------------------------------------
def write_stats_md(path, s):
    f = s["sft"]["funnel"]
    lines = [
        "# Week8 Day40 数据统计报告",
        "",
        "> 本文件由 `Week8/scripts/step1_data_prep.py` 自动生成，勿手改。",
        "",
        f"- 生成时间：{s['timestamp']}　脚本版本：`step1_data_prep.py v{s['script_version']}`",
        f"- 原始数据：`{s['source']}`",
        f"- 分词后端：`{s['tokenizer_backend']}`　cutoff：{s['max_tokens']} token",
        f"- 划分：train:val = {1 - s['val_ratio']:.0%}:{s['val_ratio']:.0%}，seed={s['seed']}"
        + ("　**（--quick 冒烟模式，非完整数据）**" if s["quick"] else ""),
        f"- 总耗时：{s['elapsed_sec']} 秒",
        "",
        "## 一、SFT 数据清洗漏斗",
        "",
        "| 阶段 | 样本数 | 说明 |",
        "|---|---|---|",
        f"| 0 原始 | {f['0_raw']} | `alpaca_all.json` 直接读入 |",
        f"| 1 文本清洗 | {f['1_text_cleaned']} | HTML 标签 / 控制字符 / 空白规范；"
        f"改动 {s['sft']['modified_by_clean_text']} 条，不丢样本 |",
        f"| 2 空值过滤 | {f['2_drop_empty']} | instruction 或 output 为空则丢 |",
        f"| 3 长度处理 | {f['3_length_handled']} | 截断 output {s['sft']['truncated']} 条；"
        f"instruction 本身超长丢弃 {s['sft']['dropped_too_long']} 条 |",
        f"| 4 模糊去重 | {f['4_deduped']} | SimHash(64bit)+LSH，命中重复 "
        f"**{s['sft']['dropped_dup']}** 条 |",
        f"| 5 划分 | train {f['5_train']} / val {f['5_val']} | 随机 9:1，seed={s['seed']} |",
        "",
        "## 二、长度分布",
        "",
        "| 集合 | 口径 | n | min | median | p90 | max | mean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in ("train", "val"):
        for unit in ("token", "char"):
            d = s["sft"][f"len_{name}_{unit}"]
            lines.append(f"| {name} | {unit} | {d['n']} | {d['min']} | {d['median']} | "
                         f"{d['p90']} | {d['max']} | {d['mean']} |")
    lines += ["", "## 三、来源分布", "",
              "| 来源 | 原始 | train | val |", "|---|---|---|---|"]
    for k in s["sft"]["source_raw"]:
        lines.append(f"| {k} | {s['sft']['source_raw'][k]} | "
                     f"{s['sft']['source_train'].get(k, 0)} | "
                     f"{s['sft']['source_val'].get(k, 0)} |")

    if s.get("dpo"):
        df = s["dpo"]["funnel"]
        lines += ["", "## 四、DPO 偏好数据", "",
                  "| 阶段 | 样本数 |", "|---|---|",
                  f"| 0 原始 | {df['0_raw']} |",
                  f"| 1 清洗（去空 / 去 chosen==rejected） | {df['1_cleaned']} |",
                  f"| 2 按 prompt 去重 | {df['2_deduped']} |",
                  f"| 3 划分 | train {df['3_train']} / val {df['3_val']} |",
                  "",
                  f"- 去重命中：**{s['dpo']['dropped_dup']}** 条",
                  "",
                  "| 集合 | 口径 | n | min | median | p90 | max | mean |",
                  "|---|---|---|---|---|---|---|---|"]
        for name in ("train", "val"):
            d = s["dpo"].get(f"len_{name}_token")
            if d:
                lines.append(f"| dpo_{name} | token | {d['n']} | {d['min']} | {d['median']} | "
                             f"{d['p90']} | {d['max']} | {d['mean']} |")

    lines += ["", "## 五、耗时（秒）", "", "| 阶段 | 秒 |", "|---|---|"]
    for k, v in s["timings"].items():
        lines.append(f"| {k} | {v} |")

    lines += ["", "## 六、产物", "",
              "| 文件 | 格式 | 用途 |", "|---|---|---|",
              "| `train.json` / `val.json` | Alpaca | SFT 训练 / 验证（sft_best.yaml 直接吃） |",
              "| `train_sharegpt.json` / `val_sharegpt.json` | ShareGPT | 同一批样本的多轮编码，备用 |",
              "| `dpo_train.json` / `dpo_val.json` | ShareGPT + ranking | DPO 训练 / 验证 |",
              "| `dataset_info.json` | — | LLaMA-Factory 注册表，6 个数据集 |",
              "",
              "> ★ 注意：`sft_best.yaml` / `dpo_best.yaml` 里**没有** `val_size`。",
              "> 验证集由本脚本显式切好，通过 `eval_dataset:` 传给 LLaMA-Factory。",
              "> 两者不能并存：同时写 LF 会抛 "
              "`Cannot specify val_size if eval_dataset is not None`；",
              "> 只留 `val_size` 则本脚本切的 val 集完全不被使用，LF 会在 train 上"
              "另切一刀（静默的二次划分）。",
              ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Week8 Day40 数据准备：清洗 / 去重 / 双格式转换 / 9:1 划分 / 统计报告")
    ap.add_argument("--source", default="Week2/data/unified/alpaca_all.json",
                    help="原始 Alpaca 池（默认复用 Week2 download_data.py 的产物）")
    ap.add_argument("--dpo-source", default="Week4/data/dpo/dpo_pairs.json",
                    help="原始偏好对（默认复用 Week4 产物）；文件不存在则跳过 DPO 部分")
    ap.add_argument("--out-dir", default="Week8/data")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例，任务书要求 9:1")
    ap.add_argument("--seed", type=int, default=42, help="划分随机种子，与 Week3/Week4 一致")
    ap.add_argument("--tokenizer", default="models/Qwen2.5-3B-Instruct",
                    help="HF 分词器路径；不可用时自动回退 tiktoken / 字符启发式")
    ap.add_argument("--max-tokens", type=int, default=cp.MAX_TOKENS,
                    help="长度上限，必须与 sft_best.yaml 的 cutoff_len 一致")
    ap.add_argument("--quick", type=int, default=0, metavar="N",
                    help="冒烟模式：只取前 N 条（0=全量）")
    ap.add_argument("--no-dpo", action="store_true", help="只处理 SFT 数据")
    args = ap.parse_args()

    t_start = time.time()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 分词器路径转绝对：本脚本可能被 step2 从任意 cwd 调用，相对路径会找不到。
    tok_path = args.tokenizer
    if tok_path and not Path(tok_path).is_absolute():
        tok_path = str(ROOT / tok_path)
    count_tok, trunc_tok, backend = cp.build_token_counter(tok_path)
    print(f"[step1] 分词后端: {backend}")

    timings = {}
    train, val, funnel, sft_stats = prepare_sft(args, count_tok, trunc_tok, timings)

    # 双格式落盘（任务书 40.1 的"格式转换（Alpaca/ShareGPT）"）
    write_json(out_dir / "train.json", train)
    write_json(out_dir / "val.json", val)
    write_json(out_dir / "train_sharegpt.json", [to_sharegpt(r) for r in train])
    write_json(out_dir / "val_sharegpt.json", [to_sharegpt(r) for r in val])

    def _lens(recs, unit):
        if unit == "token":
            return [count_tok(r["instruction"] + r.get("input", "") + r["output"]) for r in recs]
        return [len(r["instruction"]) + len(r.get("input", "")) + len(r["output"]) for r in recs]

    sft_stats["funnel"] = funnel
    for name, recs in (("train", train), ("val", val)):
        for unit in ("token", "char"):
            sft_stats[f"len_{name}_{unit}"] = dist(_lens(recs, unit))

    dpo_stats = None
    if not args.no_dpo:
        dpo_train, dpo_val, dpo_stats = prepare_dpo(args, count_tok, timings)
        if dpo_train is not None:
            write_json(out_dir / "dpo_train.json", dpo_train)
            write_json(out_dir / "dpo_val.json", dpo_val)

    write_json(out_dir / "dataset_info.json", build_dataset_info())

    stats = {
        "script_version": SCRIPT_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": args.source,
        "dpo_source": args.dpo_source,
        "tokenizer_backend": backend,
        "max_tokens": args.max_tokens,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "quick": args.quick,
        "out_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "sft": sft_stats,
        "dpo": dpo_stats,
        "timings": timings,
        "elapsed_sec": round(time.time() - t_start, 2),
    }
    deliv = ROOT / "Week8" / "deliverables"
    write_json(deliv / "data_stats.json", stats)
    write_stats_md(deliv / "data_stats.md", stats)

    print("\n=== SFT 清洗漏斗 ===")
    for k, v in funnel.items():
        print(f"  {k:<20} {v}")
    print(f"  截断 {sft_stats['truncated']}　超长丢弃 {sft_stats['dropped_too_long']}"
          f"　去重丢弃 {sft_stats['dropped_dup']}")
    if dpo_stats:
        print("=== DPO 漏斗 ===")
        for k, v in dpo_stats["funnel"].items():
            print(f"  {k:<20} {v}")
    print(f"\n产物目录: {out_dir}")
    print(f"统计报告: {deliv / 'data_stats.json'} / .md")
    print(f"总耗时: {stats['elapsed_sec']}s")
    if not args.quick and funnel["5_train"] < 1000:
        print("⚠️ 训练集不足 1000 条，请检查上游数据是否完整")


if __name__ == "__main__":
    main()
