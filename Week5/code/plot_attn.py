"""Day24 第二步:把 attn_hook.py 存下的注意力子块画成热力图。

三种产出:
  1. 单锚点热力图   原图 + 叠加图,回答"生成这个词时在看图的哪里"
  2. 逐层演化图     同一个锚点在多层的注意力对比,回答"浅层看什么、深层看什么"
  3. 图像注意力占比 每个生成 token 有多少注意力落在图像上(定量指标,进周报表格)

关键实现选择:
  * **多头聚合用 max 而不是 mean**。不同 head 分工不同(有的看边缘、有的看语义),
    取均值会把信号糊成一片平均分布,热力图看不出东西。
  * 归一化按"单张图自身最大值"做,只比较空间分布,不跨层比较绝对强度
    (绝对强度受 softmax 分母影响,跨层不可比)。

用法:
    .venv-vlm\\Scripts\\python.exe Week5/code/plot_attn.py \\
        --npz Week5/data/attn_npz/qwen_01_table.npz --anchor 0.790 --layer 20
    .venv-vlm\\Scripts\\python.exe Week5/code/plot_attn.py \\
        --npz Week5/data/attn_npz/qwen_01_table.npz --anchor 0.790 --evolution
    .venv-vlm\\Scripts\\python.exe Week5/code/plot_attn.py \\
        --npz Week5/data/attn_npz/qwen_01_table.npz --list-tokens
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "Week5" / "data" / "images"
OUT_DIR = ROOT / "Week5" / "deliverables" / "attn"

for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=_f).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False


def load_npz(path: Path) -> tuple[dict, dict[int, np.ndarray]]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    attn = {int(k.split("_")[1]): z[k] for k in z.files if k.startswith("layer_")}
    return meta, attn


def find_anchor(tokens: list[str], anchor: str) -> tuple[int, int]:
    """在生成的 token 序列里定位锚点文本,返回 [起, 止) 的 token 下标区间。

    锚点常常跨多个 token(比如 "0.790" 可能被切成 "0", ".", "790"),
    所以按拼接串做子串匹配,而不是逐 token 相等比较。
    """
    joined, spans = "", []
    for i, t in enumerate(tokens):
        spans.append((len(joined), len(joined) + len(t), i))
        joined += t
    pos = joined.find(anchor)
    if pos < 0:
        raise SystemExit(
            f"[中止] 生成结果里找不到锚点 {anchor!r}。\n"
            f"用 --list-tokens 看看实际生成了哪些 token。\n"
            f"完整答案:{joined[:400]}")
    end = pos + len(anchor)
    hit = [i for (a, b, i) in spans if a < end and b > pos]
    return hit[0], hit[-1] + 1


def attn_map(a: np.ndarray, t0: int, t1: int, rows: int, cols: int,
             head: str | int = "max") -> np.ndarray:
    """a: [heads, n_gen, n_img] -> 归一化的 (rows, cols) 二维图。"""
    sub = a[:, t0:t1, :]                      # [heads, span, n_img]
    sub = sub.mean(axis=1)                    # 锚点跨多 token 时对 span 取均值
    if head == "max":
        v = sub.max(axis=0)
    elif head == "mean":
        v = sub.mean(axis=0)
    else:
        v = sub[int(head)]
    n = rows * cols
    if v.shape[0] < n:                        # 兜底:padding 或网格推断偏差
        v = np.pad(v, (0, n - v.shape[0]))
    grid = v[:n].reshape(rows, cols)
    m = grid.max()
    return grid / m if m > 0 else grid


def upsample(grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """(rows, cols) -> 原图尺寸 (width, height),双三次插值。"""
    im = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    return np.asarray(im.resize(size, Image.BICUBIC), dtype=np.float32) / 255.0


def overlay(ax, img: Image.Image, grid: np.ndarray, title: str) -> None:
    ax.imshow(img)
    ax.imshow(upsample(grid, img.size), cmap="jet", alpha=0.45,
              extent=(0, img.size[0], img.size[1], 0))
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def plot_single(meta: dict, attn: dict, anchor: str, layer: int, head: str | int) -> Path:
    tokens = meta["gen_tokens"]
    t0, t1 = find_anchor(tokens, anchor)
    img = Image.open(IMG_DIR / meta["image"]).convert("RGB")
    grid = attn_map(attn[layer], t0, t1, meta["grid_rows"], meta["grid_cols"], head)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(img)
    axes[0].set_title(f"原图 {meta['image']}  {img.size[0]}×{img.size[1]}", fontsize=11)
    axes[0].axis("off")
    overlay(axes[1], img, grid,
            f"第 {layer}/{meta['n_layers']} 层  head={head}  锚点 = {anchor!r}")
    fig.suptitle(f"{meta['display']}　跨模态注意力（self-attn 的 text→image 子块）\n"
                 f"视觉 token {meta['n_img']} 个 → {meta['grid_rows']}×{meta['grid_cols']} 网格",
                 fontsize=13)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in anchor)[:20]
    out = OUT_DIR / f"{meta['model']}_{Path(meta['image']).stem}_L{layer}_{safe}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] {out}")
    return out


def plot_evolution(meta: dict, attn: dict, anchor: str, head: str | int) -> Path:
    tokens = meta["gen_tokens"]
    t0, t1 = find_anchor(tokens, anchor)
    img = Image.open(IMG_DIR / meta["image"]).convert("RGB")
    layers = sorted(attn)

    n = len(layers) + 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.2))
    axes[0].imshow(img)
    axes[0].set_title("原图", fontsize=11)
    axes[0].axis("off")
    for ax, l in zip(axes[1:], layers):
        grid = attn_map(attn[l], t0, t1, meta["grid_rows"], meta["grid_cols"], head)
        overlay(ax, img, grid, f"第 {l} 层")
    fig.suptitle(f"{meta['display']}　注意力逐层演化　锚点 = {anchor!r}　"
                 f"(共 {meta['n_layers']} 层, head={head})", fontsize=13)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in anchor)[:20]
    out = OUT_DIR / f"{meta['model']}_{Path(meta['image']).stem}_layers_{safe}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[写出] {out}")
    return out


def list_tokens(meta: dict, attn: dict, layer: int, top: int = 25) -> None:
    """列出每个生成 token 的「图像注意力占比」= 落在图像 token 上的注意力之和。

    这是一个定量指标:名词/数字 token 的占比通常显著高于虚词,
    可以直接进周报,证明模型确实在"看图说话"而不是在背语言模型先验。
    """
    a = attn[layer]                          # [heads, n_gen, n_img]
    ratio = a.mean(axis=0).sum(axis=1)       # 对 head 取均值,再对图像列求和
    tokens = meta["gen_tokens"]
    n = min(len(tokens), ratio.shape[0])
    print(f"\n=== {meta['display']} 第 {layer} 层：每个生成 token 的图像注意力占比 ===")
    print(f"{'idx':>4} {'token':<16} {'图像注意力占比':>12}")
    for i in range(n):
        bar = "█" * int(ratio[i] * 40)
        print(f"{i:>4} {tokens[i]!r:<16} {ratio[i]:>12.3f} {bar}")
    order = np.argsort(-ratio[:n])[:top]
    print(f"\n占比最高的 {top} 个 token:")
    print("  " + "  ".join(f"{tokens[i]!r}={ratio[i]:.2f}" for i in order))
    print(f"\n全序列均值 {ratio[:n].mean():.3f}　最大 {ratio[:n].max():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--anchor", help="要可视化的生成文本片段,如 0.790")
    ap.add_argument("--layer", type=int, help="默认取抓取层里居中的一层")
    ap.add_argument("--head", default="max", help="max(默认) | mean | 具体 head 下标")
    ap.add_argument("--evolution", action="store_true", help="画逐层演化图")
    ap.add_argument("--list-tokens", action="store_true", help="列出每 token 的图像注意力占比")
    args = ap.parse_args()

    meta, attn = load_npz(Path(args.npz))
    head = args.head if args.head in ("max", "mean") else int(args.head)
    layer = args.layer if args.layer is not None else sorted(attn)[len(attn) // 2]
    if layer not in attn:
        raise SystemExit(f"[中止] npz 里没有第 {layer} 层,只有 {sorted(attn)}")

    print(f"{meta['display']} / {meta['image']}　抓取层 {sorted(attn)}　"
          f"网格 {meta['grid_rows']}×{meta['grid_cols']}　heads {meta['n_heads']}")
    print(f"问题:{meta['question']}")
    print(f"回答:{meta['answer'][:200]}\n")

    if args.list_tokens:
        list_tokens(meta, attn, layer)
    if args.anchor:
        plot_single(meta, attn, args.anchor, layer, head)
        if args.evolution:
            plot_evolution(meta, attn, args.anchor, head)
    elif not args.list_tokens:
        raise SystemExit("[中止] 需要 --anchor 或 --list-tokens")
