"""Day24 第一步:用 forward hook 抓取指定层的注意力,切出 text→image 子块并存盘。

## 为什么不是 Cross-Attention

任务书 24.1 写"提取 Cross-Attention 权重"。但 Qwen2.5-VL 和 Gemma-4 都是
**soft-token 注入范式**:视觉特征被 projector 映射后,直接占据文本序列里
`<|image_pad|>` / `<|image|>` 占位符的位置,之后全程只有 self-attention。
这两个模型的模块树里**没有任何 cross_attn 层**(已核对 transformers 5.14.1 源码)。

所以正确做法是取 self-attention 矩阵的子块:

    A[layer][0, head, t, img_start:img_end]

含义:生成第 t 个 token 时,注意力有多少落在图像的哪些位置上。
这在语义上就是"跨模态注意力",只是实现上寄生在 self-attention 里。

## 两个必踩的坑

1. **必须 attn_implementation="eager"**。SDPA 和 FlashAttention 从不显式构造
   注意力矩阵(这正是它们快且省显存的原因),`output_attentions=True` 在它们下面
   会返回 None 或直接报错。本机装了 flash-attn,默认走 sdpa,不改就拿不到权重。

2. **不要在 generate() 循环里抓**。带 KV cache 时每步的注意力形状是
   [B, heads, 1, past+1],拼接起来很容易错位。这里用两段式:
   先正常快速生成答案 → 再把 (prompt + 答案) 拼成完整序列做**一次** eager 前向
   (teacher forcing),等价于生成时的注意力,而且事后可以任选锚点 token。

用法:
    .venv-vlm\\Scripts\\python.exe Week5/code/attn_hook.py --model qwen \\
        --image 01_table.png --layers 5 14 20 27 \\
        --question "这张表里 eval acc 最高的是哪一行?它的 eval margin 是多少?"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (  # noqa: E402
    ROOT, SPECS, build_messages, image_token_grid, image_token_span,
    load_vlm, prepare_inputs,
)

IMG_DIR = ROOT / "Week5" / "data" / "images"
ATTN_DIR = ROOT / "Week5" / "deliverables" / "attn"
NPZ_DIR = ROOT / "Week5" / "data" / "attn_npz"


def text_layers(model) -> torch.nn.ModuleList:
    """两个模型的解码层都在 model.model.language_model.layers,但留出回退路径。"""
    for path in ("model.language_model.layers", "language_model.model.layers",
                 "model.layers", "language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, torch.nn.ModuleList) and len(obj):
                return obj
        except AttributeError:
            continue
    raise RuntimeError("找不到解码器层列表,模型结构可能变了")


@torch.no_grad()
def capture(model_key: str, image_name: str, question: str,
            layers: list[int], max_new_tokens: int = 128) -> dict:
    img_path = IMG_DIR / image_name
    if not img_path.exists():
        raise FileNotFoundError(f"{img_path} 不存在")
    with Image.open(img_path) as im:
        img_size = im.size  # (width, height)

    # ---- 第一段:eager 加载 + 正常生成答案 ----
    # 同一个实例既生成又做前向,避免两次加载 16GB 权重。
    vlm = load_vlm(model_key, attn_impl="eager")
    lm_layers = text_layers(vlm.model)
    n_layers = len(lm_layers)
    layers = sorted({(l if l >= 0 else n_layers + l) for l in layers if -n_layers <= l < n_layers})
    print(f"[{model_key}] 解码层共 {n_layers} 层,本次抓取 {layers}")

    messages = build_messages(model_key, img_path, question)
    inputs = prepare_inputs(vlm, messages)
    n_prompt = int(inputs["input_ids"].shape[-1])
    img_start, img_end = image_token_span(vlm, inputs)
    rows, cols = image_token_grid(vlm, inputs, img_size)
    n_img = img_end - img_start
    print(f"[{model_key}] 提示 {n_prompt} token;图像 token 区间 [{img_start},{img_end}) "
          f"共 {n_img} 个,还原成 {rows}×{cols}={rows * cols} 网格")
    if rows * cols != n_img:
        print(f"⚠️  网格 {rows}×{cols} 与图像 token 数 {n_img} 不一致,热力图可能错位")

    out_ids = vlm.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = out_ids[0][n_prompt:]
    answer = vlm.processor.decode(gen_ids, skip_special_tokens=True).strip()
    tok_strs = [vlm.processor.tokenizer.decode([int(t)]) for t in gen_ids]
    print(f"[{model_key}] 生成 {len(gen_ids)} token:{answer[:150]}\n")

    # ---- 第二段:整条序列做一次 eager 前向,hook 抓目标层 ----
    grabbed: dict[int, np.ndarray] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_mod, _args, output):
            # Qwen2_5_VLAttention / Gemma4TextAttention 的 forward 都返回
            # (attn_output, attn_weights);eager 下 attn_weights 形状
            # [B, heads, q_len, kv_len]
            w = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
            if w is None:
                return
            # 只留 "答案位置 → 图像位置" 的子块,否则 [heads, 2000, 2000] 会吃掉几百 MB
            sub = w[0, :, n_prompt - 1:, img_start:img_end]
            grabbed[idx] = sub.detach().to(torch.float32).cpu().numpy()
        return hook

    for i in layers:
        handles.append(lm_layers[i].self_attn.register_forward_hook(make_hook(i)))

    full_ids = out_ids[:, :]
    fwd = {"input_ids": full_ids,
           "attention_mask": torch.ones_like(full_ids)}
    for k in ("pixel_values", "image_grid_thw", "image_position_ids",
              "num_soft_tokens_per_image", "token_type_ids", "mm_token_type_ids"):
        if k in inputs:
            v = inputs[k]
            if k in ("token_type_ids", "mm_token_type_ids"):
                # 这两个是逐 token 的,长度必须补到完整序列(新生成的都是文本=0)
                pad = torch.zeros((v.shape[0], full_ids.shape[1] - v.shape[1]),
                                  dtype=v.dtype, device=v.device)
                v = torch.cat([v, pad], dim=1)
            fwd[k] = v

    try:
        vlm.model(**fwd, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    missing = [i for i in layers if i not in grabbed]
    if missing:
        raise RuntimeError(
            f"第 {missing} 层没抓到注意力权重。最常见原因:attn_implementation 不是 'eager'。"
            f"当前={vlm.attn_impl}")

    meta = {
        "model": model_key, "display": SPECS[model_key].display,
        "image": image_name, "image_size": list(img_size),
        "question": question, "answer": answer,
        "n_layers": n_layers, "layers": layers,
        "n_prompt": n_prompt, "img_start": img_start, "img_end": img_end,
        "n_img": n_img, "grid_rows": rows, "grid_cols": cols,
        "n_heads": int(next(iter(grabbed.values())).shape[0]),
        "gen_tokens": tok_strs,
    }
    del vlm
    torch.cuda.empty_cache()
    return {"meta": meta, "attn": grabbed}


def save(pack: dict) -> Path:
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    m = pack["meta"]
    stem = f"{m['model']}_{Path(m['image']).stem}"
    npz = NPZ_DIR / f"{stem}.npz"
    np.savez_compressed(npz, meta=json.dumps(m, ensure_ascii=False),
                        **{f"layer_{i}": a for i, a in pack["attn"].items()})
    print(f"[写出] {npz}  ({npz.stat().st_size / 1024 ** 2:.1f} MB)")
    return npz


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SPECS), required=True)
    ap.add_argument("--image", required=True, help="Week5/data/images/ 下的文件名")
    ap.add_argument("--question", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 14, 20, 27],
                    help="解码层下标,支持负数(-1 = 最后一层)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    ATTN_DIR.mkdir(parents=True, exist_ok=True)
    pack = capture(args.model, args.image, args.question, args.layers, args.max_new_tokens)
    save(pack)
    print("\n下一步:python Week5/code/plot_attn.py --npz "
          f"Week5/data/attn_npz/{args.model}_{Path(args.image).stem}.npz --anchor <某个词>")
