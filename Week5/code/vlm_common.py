"""Week5 两个 VLM 的统一加载/推理封装。

设计目标:Day23(批量推理)、Day24(注意力可视化)、Day25(幻觉检测)、Day26(微调前后对比)
四个脚本共用同一套加载逻辑,保证对比公平(同 dtype、同显存策略、同生成参数)。

两个模型的关键差异(这是本周要理解的核心):

  Qwen2.5-VL-7B-Instruct  中国 / Apache-2.0
    ViT 原生动态分辨率 → 视觉 token 数随图片尺寸变化(≈ H/28 × W/28,2×2 merge)
    视觉 token 直接插入文本序列的 <|image_pad|> 位置 → 全程只有 self-attention
    位置编码 M-RoPE(时间/高度/宽度三维)

  gemma-4-E4B-it          美国 / Apache-2.0
    ~150M vision encoder(16 层 / 768 维 / 12 头),学习式 2D 位置 + 多维 RoPE
    保持原始宽高比,但 soft token 预算固定可选 70/140/280/560/1120
    soft token 同样插入文本序列 → 也是 self-attention

  → 两者是同一范式(projector/soft-token 注入)的两种极端:动态 token 数 vs 固定预算。
    这正好是 Day24 注意力可视化可以直接对比的一条轴。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

# Qwen 的像素预算:1280 个 28×28 块 ≈ 100 万像素。
# 不设这个参数的话,一张 4K 截图能吃掉 8GB 显存(视觉 token 上万)。
QWEN_MIN_PIXELS = 256 * 28 * 28
QWEN_MAX_PIXELS = 1280 * 28 * 28

# Gemma 4 的 soft token 预算,官方支持 70/140/280/560/1120,默认 280,
# 通过 Gemma4Processor(image_seq_length=...) 设置(已在 transformers 5.14.1 上核实签名)。
# 表格/UI/手写这类要看细节的图必须提到 560 或 1120,否则细节被池化掉、OCR 必错。
GEMMA_TOKEN_BUDGET = 560

GEN_KWARGS = dict(max_new_tokens=512, do_sample=False)  # 贪心解码:实验要可复现


@dataclass
class VLMSpec:
    key: str
    local_dir: str
    display: str
    origin: str
    template: str          # LLaMA-Factory 模板名,Day26 微调用
    image_token: str


SPECS: dict[str, VLMSpec] = {
    "qwen": VLMSpec("qwen", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct",
                    "中国 / 阿里", "qwen2_vl", "<|image_pad|>"),
    "gemma": VLMSpec("gemma", "gemma-4-E4B-it", "gemma-4-E4B-it",
                     "美国 / Google", "gemma4", "<|image|>"),
}


@dataclass
class LoadedVLM:
    spec: VLMSpec
    model: object
    processor: object
    attn_impl: str
    load_seconds: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def device(self):
        return self.model.device


# ---------------------------------------------------------------- 加载
def model_path(key: str) -> Path:
    p = MODELS / SPECS[key].local_dir
    hint = f"先跑:.venv\\Scripts\\python.exe Week5/code/download_vlm.py --only {key}"
    if not (p / "config.json").exists():
        raise FileNotFoundError(f"{p} 不存在或未下载完成。{hint}")
    # 光有 config.json 不代表下载完了:必须确认没有残留的 .incomplete 分片,
    # 否则 from_pretrained 会抛出难以定位的 safetensors 反序列化错误。
    partial = sorted(x.name for x in p.glob("*.incomplete"))
    if partial:
        raise FileNotFoundError(
            f"{p} 下载未完成,仍有 {len(partial)} 个分片在传输中(如 {partial[0]})。{hint}")
    if not any(p.glob("*.safetensors")):
        raise FileNotFoundError(f"{p} 里没有 safetensors 权重文件。{hint}")
    return p


def load_vlm(key: str, attn_impl: str = "sdpa", load_4bit: bool = False,
             path_override: str | Path | None = None,
             adapter: str | Path | None = None) -> LoadedVLM:
    """attn_impl:Day23/25 用 'sdpa'(快);Day24 必须用 'eager',否则拿不到注意力矩阵。

    SDPA / FlashAttention 从不显式构造注意力矩阵(这正是它们快且省显存的原因),
    因此 output_attentions=True 在这两种实现下会返回 None 或直接报错。
    """
    from transformers import AutoProcessor

    spec = SPECS[key]
    # path_override:Day26 要加载合并后的微调模型;processor 仍从同一目录读,
    # 因为 LLaMA-Factory export 会把 processor 配置一起写出去。
    path = Path(path_override) if path_override else model_path(key)
    kwargs: dict = dict(dtype=torch.bfloat16, attn_implementation=attn_impl)
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    else:
        kwargs["device_map"] = "cuda:0"

    # 一律用 perf_counter(单调时钟)测耗时,不用 time.time()。
    # 本机 w32time 未同步(Source: Local CMOS Clock),实测跑 Day23 时系统时钟
    # 向前跳了 10.6 小时,用挂钟测出来的单条延迟直接变成 38088 秒。
    t0 = time.perf_counter()
    if key == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, **kwargs)
        processor = AutoProcessor.from_pretrained(
            path, min_pixels=QWEN_MIN_PIXELS, max_pixels=QWEN_MAX_PIXELS)
    else:
        from transformers import Gemma4ForConditionalGeneration
        model = Gemma4ForConditionalGeneration.from_pretrained(path, **kwargs)
        # soft token 预算真正生效的地方是 image_processor.max_soft_tokens
        # (Gemma4Processor.image_seq_length 只是文本侧的默认占位数)。
        # 源码里 max_patches = max_soft_tokens * pooling_kernel_size**2,
        # 图片被等比缩放到 patch 数不超过这个预算,且边长是 48 的倍数(3×16)。
        processor = AutoProcessor.from_pretrained(
            path, image_seq_length=GEMMA_TOKEN_BUDGET, max_soft_tokens=GEMMA_TOKEN_BUDGET)
        processor.image_processor.max_soft_tokens = GEMMA_TOKEN_BUDGET
        processor.image_seq_length = GEMMA_TOKEN_BUDGET
        assert processor.image_processor.max_soft_tokens == GEMMA_TOKEN_BUDGET

    if adapter:  # 不合并直接挂 LoRA:省一次 16GB 导出,适合快速验证
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    load_s = time.perf_counter() - t0

    return LoadedVLM(spec=spec, model=model, processor=processor,
                     attn_impl=attn_impl, load_seconds=load_s,
                     meta={"4bit": load_4bit, "path": str(path),
                           "adapter": str(adapter) if adapter else None})


# ---------------------------------------------------------------- 消息构造
def build_messages(key: str, image: str | Path | None, question: str,
                   system: str | None = None) -> list[dict]:
    """两个模型的 chat content 结构不同:
    Qwen 用 {"type":"image","image": <路径/PIL>};Gemma4 走 transformers 5.x 通用
    多模态模板,用 {"type":"image","url": <路径>}。
    """
    content: list[dict] = []
    if image is not None:
        p = str(Path(image).resolve())
        content.append({"type": "image", "image": p} if key == "qwen"
                       else {"type": "image", "url": p})
    content.append({"type": "text", "text": question})

    msgs = []
    if system:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
    msgs.append({"role": "user", "content": content})
    return msgs


def prepare_inputs(vlm: LoadedVLM, messages: list[dict]):
    key = vlm.spec.key
    proc = vlm.processor
    if key == "qwen":
        from qwen_vl_utils import process_vision_info
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = proc(text=[text], images=images, videos=videos,
                      padding=True, return_tensors="pt")
    else:
        inputs = proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
    return inputs.to(vlm.model.device)


def image_token_id(vlm: LoadedVLM) -> int | None:
    tok = vlm.processor.tokenizer
    tid = tok.convert_tokens_to_ids(vlm.spec.image_token)
    if tid is None or tid == getattr(tok, "unk_token_id", -1):
        tid = getattr(vlm.model.config, "image_token_id", None)  # 回退到 config 声明
    return tid


def count_image_tokens(vlm: LoadedVLM, inputs) -> int:
    """统计这张图占了多少个 token —— 显存和延迟的第一解释变量。"""
    tid = image_token_id(vlm)
    if tid is None:
        return -1
    return int((inputs["input_ids"] == tid).sum().item())


def image_token_span(vlm: LoadedVLM, inputs) -> tuple[int, int]:
    """图像 token 在文本序列里的 [起, 止) 区间。Day24 取注意力子块要用。"""
    tid = image_token_id(vlm)
    pos = (inputs["input_ids"][0] == tid).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        raise RuntimeError(f"输入序列里找不到图像占位符 token(id={tid})")
    return int(pos[0].item()), int(pos[-1].item()) + 1


def image_token_grid(vlm: LoadedVLM, inputs, image_size: tuple[int, int]) -> tuple[int, int]:
    """把一维图像 token 序列还原成二维网格 (行, 列) —— 画热力图的前提。

    image_size: PIL 的 (width, height) 原始尺寸。
    """
    if vlm.spec.key == "qwen":
        # image_grid_thw 是 patch 单位的 (t, h, w),patch=14;
        # merger 做 2×2 合并,所以 token 网格是 (h//2, w//2)。
        t, h, w = (int(x) for x in inputs["image_grid_thw"][0])
        merge = int(getattr(vlm.model.config.vision_config, "spatial_merge_size", 2))
        return h // merge, w // merge

    # Gemma4:等比缩放到 patch 数 ≤ max_soft_tokens×9 且边长为 48 的倍数,
    # 再按 3×3 池化 → soft token 网格 = (H/48, W/48)。
    from transformers.models.gemma4.image_processing_gemma4 import (
        get_aspect_ratio_preserving_size,
    )
    ip = vlm.processor.image_processor
    ps, pk = int(ip.patch_size), int(ip.pooling_kernel_size)
    th, tw = get_aspect_ratio_preserving_size(
        height=image_size[1], width=image_size[0], patch_size=ps,
        max_patches=int(ip.max_soft_tokens) * pk ** 2, pooling_kernel_size=pk)
    return th // (ps * pk), tw // (ps * pk)


# ---------------------------------------------------------------- 生成
@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    image_tokens: int
    new_tokens: int
    latency_s: float
    peak_mem_gb: float


@torch.no_grad()
def generate_from_messages(vlm: LoadedVLM, messages: list[dict], **gen_overrides) -> GenResult:
    """多轮版本:Day25 的诱导性追问(sycophancy)需要在同一张图上接着问第二轮。"""
    inputs = prepare_inputs(vlm, messages)
    n_prompt = int(inputs["input_ids"].shape[-1])
    n_img = count_image_tokens(vlm, inputs)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()   # 单调时钟,见 load_vlm 里的说明
    out = vlm.model.generate(**inputs, **{**GEN_KWARGS, **gen_overrides})
    latency = time.perf_counter() - t0

    new_ids = out[0][n_prompt:]
    text = vlm.processor.decode(new_ids, skip_special_tokens=True).strip()
    return GenResult(text=text, prompt_tokens=n_prompt, image_tokens=n_img,
                     new_tokens=int(new_ids.shape[-1]), latency_s=latency,
                     peak_mem_gb=torch.cuda.max_memory_allocated() / 1024 ** 3)


def generate(vlm: LoadedVLM, image, question: str, system: str | None = None,
             **gen_overrides) -> GenResult:
    return generate_from_messages(
        vlm, build_messages(vlm.spec.key, image, question, system), **gen_overrides)


def append_turn(messages: list[dict], assistant_text: str, next_user: str) -> list[dict]:
    """把上一轮回答和新的追问接到对话上(图片只在第一轮出现,后续轮不再重复传)。"""
    return messages + [
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        {"role": "user", "content": [{"type": "text", "text": next_user}]},
    ]


# ---------------------------------------------------------------- 参数量拆解
def param_breakdown(vlm: LoadedVLM) -> dict[str, int]:
    """把参数分成 视觉编码器 / 跨模态投影 / 语言模型 三部分。
    Day22 交付物要这张表,也是理解'图像如何被翻译进文本空间'的第一手证据。
    """
    buckets = {"vision_encoder": 0, "projector": 0, "language_model": 0,
               "audio_tower": 0, "other": 0}
    for name, p in vlm.model.named_parameters():
        n = name.lower()
        # 顺序很重要:先判投影层。Qwen 的 merger 挂在 visual 下面,
        # Gemma 的 embed_vision 名字里带 "vision.",两者都会被视觉塔的规则误吞。
        if "merger" in n or any(k in n for k in (
                "multi_modal_projector", "mm_projector", "embed_vision",
                "vision_projection", "soft_embedding")):
            buckets["projector"] += p.numel()
        elif any(k in n for k in ("audio_tower", "embed_audio", "audio_model")):
            buckets["audio_tower"] += p.numel()   # Gemma4 E4B 带音频塔,本周不用但要单列
        elif any(k in n for k in ("visual", "vision_tower", "vision_model", "vision.")):
            buckets["vision_encoder"] += p.numel()
        elif any(k in n for k in ("language_model", "model.layers", "lm_head", "embed_tokens",
                                  "per_layer", "altup", "laurel")):
            # per_layer/altup/laurel 是 Gemma4 的 Per-Layer Embeddings 架构组件,
            # 属于语言模型侧(这也是它"有效参数 4.5B / 总参数 8B"的来源)
            buckets["language_model"] += p.numel()
        else:
            buckets["other"] += p.numel()
    buckets["total"] = sum(v for k, v in buckets.items() if k != "total")
    return buckets
