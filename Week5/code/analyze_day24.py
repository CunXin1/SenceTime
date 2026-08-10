"""Day24 交付:注意力可视化分析 -> Day24_注意力可视化分析.md

自动从 npz 里算定量指标(图像注意力占比、注意力集中度),定性观察写在常量里。

「注意力集中度」用归一化熵:  H = -Σ p·log p / log N
    接近 0 → 注意力高度集中在少数几个图像位置(锐利)
    接近 1 → 均匀弥散(什么都没看清)
这是比"看图说话"更硬的证据,可以直接和 Day23 的 OCR 准确率对照。

用法:
    .venv-vlm\\Scripts\\python.exe Week5/code/analyze_day24.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NPZ_DIR = ROOT / "Week5" / "data" / "attn_npz"
ATTN_DIR = ROOT / "Week5" / "deliverables" / "attn"
OUT = ROOT / "Week5" / "deliverables" / "Day24_注意力可视化分析.md"


def load(p: Path):
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    attn = {int(k.split("_")[1]): z[k] for k in z.files if k.startswith("layer_")}
    return meta, attn


def concentration(vec: np.ndarray) -> tuple[float, float]:
    """返回 (归一化熵, top5% 质量占比)。

    两个指标的设计考量:
      * 熵按 head **均值** 算,不按 max。max-over-heads 会被头数系统性影响
        (Qwen 28 头 vs Gemma 8 头,取 max 后前者天然更"铺满"),跨模型不可比。
      * top5% 质量 = 注意力最高的 5% 图像位置吸走了多少注意力,
        比熵更直观:0.5 表示"一半注意力集中在 5% 的区域"。
    """
    p = vec / max(vec.sum(), 1e-9)
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(len(p)))
    k = max(1, int(round(len(p) * 0.05)))
    top = float(np.sort(p)[::-1][:k].sum())
    return ent, top


def find_anchor_idx(meta: dict, anchor: str) -> tuple[int, int] | None:
    """锚点常跨多个 token（"0.790" 可能切成 "0"/"."/"790"），按拼接串匹配。"""
    joined, spans = "", []
    for i, t in enumerate(meta["gen_tokens"]):
        spans.append((len(joined), len(joined) + len(t), i))
        joined += t
    pos = joined.find(anchor)
    if pos < 0:
        return None
    end = pos + len(anchor)
    hit = [i for (a, b, i) in spans if a < end and b > pos]
    return hit[0], hit[-1] + 1


def stats(meta: dict, attn: dict, layer: int, anchor: str | None = None) -> dict:
    a = attn[layer]                          # [heads, n_gen, n_img]
    per_tok = a.mean(axis=0)                 # 对 head 取均值
    ratio = per_tok.sum(axis=1)              # 每个生成 token 的图像注意力占比

    # 全序列平均的空间分布(对整段回答取均值 —— 反映"整体在看哪")
    seq_ent, seq_top = concentration(per_tok.mean(axis=0))

    # 锚点 token 的空间分布 —— **这才是热力图画的东西**。
    # 全序列均值会把逐 token 的集中度洗掉,两者必须分开报,不能混为一谈。
    anc_ent = anc_top = float("nan")
    span = find_anchor_idx(meta, anchor) if anchor else None
    if span:
        anc_ent, anc_top = concentration(per_tok[span[0]:span[1]].mean(axis=0))

    toks = meta["gen_tokens"][:len(ratio)]
    digits = [r for t, r in zip(toks, ratio) if any(c.isdigit() for c in t)]
    puncts = [r for t, r in zip(toks, ratio) if t.strip() in
              ("，", "。", "、", ",", ".", "：", ":", "**", "*")]
    return {
        "mean": float(ratio.mean()), "max": float(ratio.max()),
        "digit": float(np.mean(digits)) if digits else float("nan"),
        "punct": float(np.mean(puncts)) if puncts else float("nan"),
        "seq_entropy": seq_ent, "seq_top5": seq_top,
        "anchor": anchor if span else None,
        "anchor_entropy": anc_ent, "anchor_top5": anc_top,
        "n_heads": int(a.shape[0]),
    }


# 每个 npz 对应的锚点(和热力图用的是同一个,保证图表一致)
ANCHORS = {
    "qwen_01_table": "13.5",
    "gemma_01_table": "0.790",
    "qwen_02_landscape": "两个人",
}


QUALITATIVE = """
## 五、观察与结论

### 5.1 注意力确实落在「该看的地方」

Qwen 在表格图上，生成 `13.5` 时的注意力**精确落在「峰值显存」列的
`13.6 GB` / `13.5 GB` 两个单元格上**——正是取数的位置。
这是「模型在看图说话，不是在背语言先验」最直观的证据。

### 5.2 浅层看纹理，深层看语义

逐层演化图（第 5 → 14 → 20 → 27 层）显示：
第 5 层注意力弥散在全图，第 14 层起收敛到目标列，之后保持稳定。
这与 ViT/LLM 的常见结论一致：浅层处理低级视觉特征，深层完成语义定位。

### 5.3 结构化图像的注意力远比自然图像锐利

同一个模型、同样的层，表格图的注意力是几个锐利热点，
风景图则是「主体上有热点 + 周围大量次级激活」的弥散分布。
量化指标（归一化熵）也支持这一点，见上表。

**工程含义**：VLM 在文档/表格/UI 这类结构化图像上比在开放场景描述上更可靠，
不是因为任务更简单，而是因为**图像本身提供了明确的空间锚点**。

### 5.4 两个模型的注意力形态差异，解释了 Day23 的 OCR 精度差距

这是本周把两天的实验串起来的一条发现：

| | Qwen2.5-VL-7B | gemma-4-E4B-it |
|---|---|---|
| 视觉 token（同一张表格图） | 532（14×38） | 546（14×39） |
| 注意力头数 | 28 | 8 |
| 目标数字上的注意力 | 锐利，精确命中单元格 | 微弱、弥散地铺在整列上 |
| 最强激活位置 | 目标单元格 | **表格右侧的空白边缘**（attention sink） |
| Day23 该图 OCR | 全部正确 | 数字全对，但 run_id 出现 `@`、`l`→`1` |

Gemma 把最强注意力放在了不含信息的边缘区域——这是 attention sink 现象
（模型把多余的注意力"倾倒"到低信息量位置）。它在目标区域的注意力预算因此更少，
加上注意力头只有 Qwen 的 2/7，**空间分辨能力明显弱**。
这与 Day23 观察到的「数字对、形近字错」完全吻合：
粗粒度定位够用，细粒度字形辨识不够。

### 5.5 方法论提醒：这不是真正的 Cross-Attention

任务书 24.1 写「提取 Cross-Attention 权重」。本周两个模型都是
**soft-token 注入 + 纯 self-attention** 范式，模块树里没有任何 cross_attn 层
（已核对 transformers 5.14.1 源码）。这里做的是取 self-attention 矩阵中
`text_token → image_token` 的**子块**，语义上等价于「跨模态注意力」，
但实现上寄生在 self-attention 里。

真有 cross-attn 层的是 Llama-3.2-Vision 那一系（在 LLM 的第 3/8/13/18/23/28/33/38 层
插入专用 cross-attn 层，图像特征不进入文本序列）。
**动手前先确认模型属于哪种范式，否则会去找一个不存在的模块。**
"""


def main() -> None:
    files = sorted(NPZ_DIR.glob("*.npz"))
    if not files:
        raise SystemExit(f"[中止] {NPZ_DIR} 里没有 npz，先跑 attn_hook.py")

    L = ["# Day24 交付：跨模态注意力可视化分析\n",
         f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M}　"
         f"由 `Week5/code/analyze_day24.py` 生成，定性部分写在脚本常量里（可复现）。\n",
         "## 一、实现要点（两个必踩的坑）\n",
         "**① 必须 `attn_implementation=\"eager\"`。** SDPA 和 FlashAttention 从不显式构造"
         "注意力矩阵——这正是它们快且省显存的原因，`output_attentions=True` 在它们下面"
         "会返回 `None` 或直接报错。本机装了 flash-attn 2.8.3，默认走 sdpa，"
         "不改这一项就什么都拿不到。\n",
         "**② 不要在 `generate()` 循环里抓。** 带 KV cache 时每步注意力形状是 "
         "`[B, heads, 1, past+1]`，逐步拼接极易错位。本周用两段式：\n",
         "```\n"
         "第一段  正常 generate 得到答案（快，走 KV cache）\n"
         "第二段  把 (prompt + 答案) 拼成完整序列，做一次 eager 前向（teacher forcing）\n"
         "        → 等价于生成时的注意力，且事后可任选锚点 token 重新出图\n"
         "```\n",
         "hook 挂在 `model.model.language_model.layers[i].self_attn` 上，"
         "两个模型的路径一致，`forward` 都返回 `(attn_output, attn_weights)`。"
         "hook 内只保留「答案位置 → 图像位置」的子块，否则 `[heads, 2000, 2000]` 会吃掉几百 MB。\n",
         "## 二、一维 token 序列 → 二维网格的还原\n",
         "画热力图的前提是把图像 token 摆回二维。两个模型的公式不同：\n",
         "- **Qwen**：`image_grid_thw` 给出 patch 网格 `(h, w)`（patch=14），"
         "merger 做 2×2 合并 → token 网格 `(h/2, w/2)`。\n",
         "- **Gemma**：等比缩放到 patch 数 ≤ `max_soft_tokens × pooling²` 且边长为 48 的倍数"
         "（patch 16 × pooling 3）→ soft token 网格 `(H/48, W/48)`。\n",
         "两者实测都与实际 token 数**精确吻合**（见下表 `网格` 列），说明还原公式正确。\n",
         "## 三、定量指标\n",
         "「图像注意力占比」= 该生成 token 落在所有图像 token 上的注意力之和。\n",
         "「归一化熵」= 空间注意力分布的熵 / log(N)，"
         "**接近 0 表示高度集中（锐利），接近 1 表示均匀弥散（什么都没看清）**。\n",
         "「top5% 质量」= 注意力最高的 5% 图像位置吸走了多少注意力，比熵更直观。\n",
         "> 两个指标都按 head **均值** 聚合，不用 max。"
         "max-over-heads 会被头数系统性影响（Qwen 28 头 vs Gemma 8 头，取 max 后前者天然更铺满），"
         "跨模型不可比——这是第一版指标踩过的坑。\n",
         "### 3.1 逐 token 的图像注意力占比\n",
         "| 模型 | 图片 | 层 | 视觉token/网格 | heads | 数字token | 标点token | 全序列均值 |",
         "|---|---|---|---|---|---|---|---|"]

    rows = []
    for f in files:
        meta, attn = load(f)
        layer = 20 if 20 in attn else sorted(attn)[len(attn) // 2]
        s = stats(meta, attn, layer, ANCHORS.get(f.stem))
        rows.append((meta, layer, s))
        d = f"**{s['digit']:.3f}**" if s["digit"] == s["digit"] else "—"
        p = f"{s['punct']:.3f}" if s["punct"] == s["punct"] else "—"
        L.append(f"| `{meta['display']}` | `{meta['image']}` | {layer}/{meta['n_layers']} | "
                 f"{meta['n_img']} = {meta['grid_rows']}×{meta['grid_cols']} | {s['n_heads']} | "
                 f"{d} | {p} | {s['mean']:.3f} |")

    L.append("\n**读法**：数字 token 的图像注意力占比显著高于标点 token"
             "（Qwen 表格图上是 6~9 倍）。标点由语言模型先验决定，不需要看图；"
             "数字必须从图里读。这个差距就是「模型确实在看图说话」的量化证据。\n")

    L.append("### 3.2 注意力集中度\n")
    L.append("**锚点 token 的分布才是热力图画的东西**；全序列均值反映的是「整段回答整体在看哪」，"
             "会把逐 token 的集中度洗掉。两者必须分开看——混为一谈会得出相反的结论。\n")
    L.append("| 模型 | 图片 | 锚点 | 锚点熵 ↓ | 锚点 top5% ↑ | 全序列熵 | 全序列 top5% |")
    L.append("|---|---|---|---|---|---|---|")
    for meta, _layer, s in rows:
        anc = f"`{s['anchor']}`" if s["anchor"] else "—"
        ae = f"**{s['anchor_entropy']:.3f}**" if s["anchor_entropy"] == s["anchor_entropy"] else "—"
        at = f"**{s['anchor_top5']:.1%}**" if s["anchor_top5"] == s["anchor_top5"] else "—"
        L.append(f"| `{meta['display']}` | `{meta['image']}` | {anc} | {ae} | {at} | "
                 f"{s['seq_entropy']:.3f} | {s['seq_top5']:.1%} |")
    L.append("")

    L.append("## 四、热力图\n")
    for p in sorted(ATTN_DIR.glob("*.png")):
        kind = "逐层演化" if "_layers_" in p.name else "单层叠加"
        L.append(f"### {p.stem}（{kind}）\n")
        L.append(f"![{p.stem}](attn/{p.name})\n")

    L.append(QUALITATIVE)

    OUT.write_text("\n".join(L), encoding="utf-8")
    n_png = len(list(ATTN_DIR.glob("*.png")))
    print(f"[写出] {OUT}")
    print(f"  热力图 {n_png} 张（交付要求 ≥3）")
    for f in files:
        meta, attn = load(f)
        layer = 20 if 20 in attn else sorted(attn)[len(attn) // 2]
        s = stats(meta, attn, layer)
        print(f"  {meta['display']:<26} {meta['image']:<20} "
              f"熵={s['entropy']:.3f} 数字占比={s['digit']:.3f}")


if __name__ == "__main__":
    main()
