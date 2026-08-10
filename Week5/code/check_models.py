"""Day22 交付:模型下载确认 + 参数量拆解 + 显存/延迟实测 + 视觉 token 数量对比。

产出 Week5/deliverables/Day22_模型下载确认.md

用法(必须用 .venv-vlm,Gemma4 需要 transformers>=5.5):
    .venv-vlm\\Scripts\\python.exe Week5/code/check_models.py
    .venv-vlm\\Scripts\\python.exe Week5/code/check_models.py --only qwen
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (  # noqa: E402
    GEMMA_TOKEN_BUDGET, QWEN_MAX_PIXELS, ROOT, SPECS,
    generate, load_vlm, model_path, param_breakdown,
)

IMG_DIR = ROOT / "Week5" / "data" / "images"
OUT_MD = ROOT / "Week5" / "deliverables" / "Day22_模型下载确认.md"
PROBE_Q = "用一句话描述这张图片的内容。"


def dir_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 ** 3


def fmt_m(n: int) -> str:
    return f"{n / 1e6:,.1f} M" if n else "—"


def probe_one(key: str, images: list[Path]) -> dict:
    print(f"\n{'=' * 72}\n加载 {SPECS[key].display}\n{'=' * 72}", flush=True)
    vlm = load_vlm(key, attn_impl="sdpa")
    print(f"[加载完成] {vlm.load_seconds:.1f}s  权重显存 {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    parts = param_breakdown(vlm)
    for k, v in parts.items():
        print(f"  {k:<18} {fmt_m(v)}")

    runs = []
    for img in images:
        r = generate(vlm, img, PROBE_Q, max_new_tokens=96)
        runs.append({
            "image": img.name,
            "image_tokens": r.image_tokens,
            "prompt_tokens": r.prompt_tokens,
            "new_tokens": r.new_tokens,
            "latency_s": round(r.latency_s, 2),
            "peak_mem_gb": round(r.peak_mem_gb, 2),
            "answer": r.text,
        })
        print(f"\n[{img.name}] 视觉token={r.image_tokens} 提示token={r.prompt_tokens} "
              f"耗时={r.latency_s:.2f}s 峰值显存={r.peak_mem_gb:.2f}GB")
        print(f"  → {r.text[:160]}")

    result = {
        "key": key,
        "display": SPECS[key].display,
        "origin": SPECS[key].origin,
        "path": str(model_path(key)),
        "size_gb": round(dir_gb(model_path(key)), 2),
        "arch": type(vlm.model).__name__,
        "dtype": str(next(vlm.model.parameters()).dtype),
        "load_seconds": round(vlm.load_seconds, 1),
        "weight_mem_gb": round(torch.cuda.memory_allocated() / 1024 ** 3, 2),
        "params": parts,
        "runs": runs,
    }

    del vlm
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


def write_md(results: list[dict]) -> None:
    L: list[str] = []
    L.append("# Day22 交付：VLM 模型下载确认\n")
    L.append(f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M}　"
             f"由 `Week5/code/check_models.py` 自动生成，手改会被覆盖。\n")

    L.append("## 一、硬件与环境\n")
    import transformers
    L.append("| 项 | 值 |\n|---|---|")
    L.append(f"| GPU | {torch.cuda.get_device_name(0)}　"
             f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB |")
    L.append(f"| torch / CUDA | {torch.__version__} / {torch.version.cuda} |")
    L.append(f"| transformers | {transformers.__version__} |")
    L.append(f"| Python / OS | {platform.python_version()} / {platform.system()} {platform.release()} |")
    L.append("| 虚拟环境 | `.venv-vlm`（独立环境，因 Gemma-4 需要 transformers>=5.5，"
             "与主环境 4.56.2 冲突） |\n")

    L.append("### 选型说明\n")
    L.append("任务书建议「8GB 显存选 2B 级 VLM」，但实测本机为 **RTX 4090 24GB**，"
             "因此上调到 7B/8B 级：\n")
    L.append("- **Qwen2.5-VL-7B-Instruct**（而非 Qwen2-VL-2B）：2B 的 OCR 幻觉过重，"
             "会污染 Day25 的幻觉率结论；7B 在 24GB 上 bf16 可直接跑。\n")
    L.append("- **gemma-4-E4B-it**（美国对照模型）：Gemma 4 五个变体中，"
             "E2B/E4B 带 ~150M vision encoder，26B-A4B/31B 带 ~550M，"
             "而 **12B Unified 是 encoder-free（无 ViT）**。"
             "需要真 ViT + 24GB 能跑 LoRA（~17GB），故选 E4B。Apache-2.0，无门禁。\n")

    L.append("## 二、下载确认\n")
    L.append("| 模型 | 来源 | 本地体积 | 模型类 | dtype | 加载耗时 | 权重显存 |\n|---|---|---|---|---|---|---|")
    for r in results:
        L.append(f"| `{r['display']}` | {r['origin']} | {r['size_gb']} GB | `{r['arch']}` | "
                 f"{r['dtype'].replace('torch.', '')} | {r['load_seconds']}s | {r['weight_mem_gb']} GB |")
    L.append("")
    for r in results:
        L.append(f"- `{r['display']}` → `{r['path']}`")
    L.append("")

    L.append("## 三、参数量拆解（图像如何被翻译进文本空间）\n")
    L.append("| 模型 | 视觉编码器 | 跨模态投影 | 语言模型 | 音频塔 | 其他 | 合计 | ViT 占比 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        p = r["params"]
        vit_pct = (p["vision_encoder"] + p["projector"]) / p["total"] if p["total"] else 0
        L.append(f"| `{r['display']}` | {fmt_m(p['vision_encoder'])} | {fmt_m(p['projector'])} | "
                 f"{fmt_m(p['language_model'])} | {fmt_m(p.get('audio_tower', 0))} | "
                 f"{fmt_m(p['other'])} | {fmt_m(p['total'])} | {vit_pct:.1%} |")
    L.append("\n**读法**：两个模型的视觉侧（编码器＋投影层）都只占总参数的很小一部分。"
             "这说明 VLM 的「看图能力」主要不是靠视觉塔的容量，而是靠投影层把视觉特征"
             "对齐到 LLM 的 embedding 空间——**这就是 Day26 敢冻结 ViT 的直接数据依据**："
             "200 条数据去动一个在几十亿图文对上预训练出来的通用特征提取器，"
             "只会破坏表征，而真正要教的「按什么格式说话」本来就在语言侧。\n")
    L.append("> Gemma-4-E4B 的音频塔本周用不到（只做图文），但它属于总参数，单列出来避免误读。"
             "它的语言模型侧还包含 Per-Layer Embeddings（PLE）组件，"
             "这也是官方标注「有效参数 4.5B / 总参数 8B」的来源。\n")

    L.append("## 四、单图推理实测（视觉 token 数 / 延迟 / 显存）\n")
    for r in results:
        L.append(f"### {r['display']}\n")
        L.append("| 图片 | 视觉 token | 提示 token | 生成 token | 耗时 | 峰值显存 |\n|---|---|---|---|---|---|")
        for x in r["runs"]:
            L.append(f"| `{x['image']}` | {x['image_tokens']} | {x['prompt_tokens']} | "
                     f"{x['new_tokens']} | {x['latency_s']}s | {x['peak_mem_gb']} GB |")
        L.append("")
        for x in r["runs"]:
            L.append(f"- **{x['image']}** → {x['answer']}")
        L.append("")

    L.append("## 五、视觉 token 策略对比（本周核心认知）\n")
    L.append(f"- **Qwen2.5-VL**：原生动态分辨率，视觉 token 数 ≈ (H/28)×(W/28)，"
             f"随图片尺寸变化；本次 `max_pixels={QWEN_MAX_PIXELS:,}`（≈100 万像素）做了上限约束。"
             f"不设这个参数，一张 4K 截图的视觉 token 会上万，直接吃掉 8GB+ 显存。\n")
    L.append(f"- **Gemma-4-E4B**：保持原始宽高比，但 soft token 预算**固定可选** "
             f"70 / 140 / 280 / 560 / 1120（默认 280，本次用 {GEMMA_TOKEN_BUDGET}）。"
             f"图再大 token 数也不涨，代价是细节丢失。\n")
    L.append("- 两者都是 **soft-token 注入 + 纯 self-attention** 范式（无 cross-attention），"
             "这一点决定了 Day24 的注意力可视化必须取 self-attention 矩阵中 "
             "`text_token → image_token` 的子块，而不是去找 cross-attn 层。\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[写出] {OUT_MD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(SPECS))
    args = ap.parse_args()

    images = sorted(p for p in IMG_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        sys.exit(f"[中止] {IMG_DIR} 里没有图片,先跑 prepare_images.py")
    print(f"探测图片 {len(images)} 张: {[p.name for p in images]}")

    keys = [args.only] if args.only else list(SPECS)
    results = [probe_one(k, images) for k in keys]

    (ROOT / "Week5" / "deliverables" / "day22_probe.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(results)
