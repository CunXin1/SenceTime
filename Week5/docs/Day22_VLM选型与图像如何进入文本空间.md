# Day22：VLM 选型 + 图像如何被"翻译"成文本空间

## 一、和前四周的根本差别

Week1–4 处理的都是 `str → token id → embedding` 这一条链路。VLM 多出一条并行链路：

```
文本： "描述这张图"  ──tokenizer──> [1234, 5678, ...] ──embed_tokens──> [n_text, d_model]
                                                                              ↘
                                                                          拼接/注入 → LLM 解码器
                                                                              ↗
图像： H×W×3 像素 ──patch化+ViT──> [n_patch, d_vision] ──projector──> [n_img, d_model]
```

**关键认知：projector 输出的向量和文本 embedding 活在同一个空间、同样的维度。**
进了解码器之后，LLM 分不清哪个是"字"哪个是"图"——这就是"图像被翻译成文本空间"的字面含义。
所以 VLM 的看图能力主要不来自视觉塔的容量，而来自 **projector 有没有把视觉特征对齐好**。

## 二、三种主流范式

| 范式 | 图像特征去哪了 | 代表模型 | 注意力 |
|---|---|---|---|
| **A. soft-token 注入**（projector） | 变成 token，**插进文本序列** | Qwen2.5-VL、Gemma 4 (E2B/E4B/26B/31B)、LLaVA、Phi-3.5-V | 只有 self-attention |
| **B. cross-attention 注入** | **不进文本序列**，通过专用 cross-attn 层注入 | Llama-3.2-Vision（11B/90B）、Flamingo、IDEFICS | self-attn + cross-attn |
| **C. encoder-free** | **没有 ViT**，raw patch 直接线性投影 | Gemma 4 **12B Unified**、Fuyu | 只有 self-attention |

本周两个模型都是 **范式 A**。这直接决定了 Day24 的做法：

> 任务书 24.1 写"提取 VLM 中间层的 Cross-Attention 权重"——
> **Qwen2.5-VL 和 Gemma 4 都没有 cross-attention 层。**
> 正确做法是取第 20 层 self-attention 矩阵中 `text_token → image_token` 的**子块**：
> `attn[0, head, text_pos, img_start:img_end]`，它的语义就是"生成这个字时在看图的哪里"。
> （范式 B 才有教科书定义的 cross-attn。Llama-3.2-Vision 在 LLM 的第 3/8/13/18/23/28/33/38 层
> 插 cross-attn 层，本周作为架构对比在周报里讨论，不下载。）

## 三、视觉 token 数怎么算（显存的第一解释变量）

### Qwen2.5-VL：动态分辨率

```
patch = 14×14；相邻 2×2 个 patch 做 merge → 一个视觉 token 覆盖 28×28 像素
视觉 token 数 ≈ ceil(H/28) × ceil(W/28)
```

| 图片 | 视觉 token 数 |
|---|---|
| 448×448 | ≈ 256 |
| 1080×1920（截图） | ≈ 2 600 |
| 2160×3840（4K） | ≈ 10 500 |

**所以 `max_pixels` 是必设参数。** 本周用 `1280 × 28 × 28 ≈ 100 万像素`（≈1280 个视觉 token）。
不设的话一张 4K 截图能吃掉 8GB+ 显存，而且注意力是 O(n²)，延迟同步爆炸。

优点：小图省算力、大图保细节。缺点：token 数不可预测，批处理和显存规划麻烦。

### Gemma 4：固定 soft token 预算

保持原始宽高比，但把图压到固定的 soft token 数，官方支持 **70 / 140 / 280 / 560 / 1120**，
默认 280。在 transformers 5.14.1 上通过 `Gemma4Processor(image_seq_length=280)` 设置。
图片尺寸必须能被 48 整除（16×3 的池化核）。

优点：token 数完全可预测，显存/延迟稳定，适合工程部署。
缺点：**大图或密集文字图会因为池化丢细节**。表格、UI、手写这类必须提到 560 或 1120，否则 OCR 必错。
——这条会成为 Day23 能力边界分析和 Day25 幻觉分析的一个直接结论。

## 四、本周两个模型的架构细节

### Qwen2.5-VL-7B-Instruct（中国 / 阿里 / Apache-2.0）
- 视觉塔：675M 参数 ViT，窗口注意力，原生动态分辨率
- 桥：MLP merger，2×2 patch 合并后投影到 LLM 的 hidden_size
- 位置编码：**M-RoPE**，把位置拆成 时间 / 高度 / 宽度 三个维度分别编码
  （所以它能天然处理视频，也能理解"左上角那个数字"这类空间指代）
- 占位符 token：`<|image_pad|>`
- 强项：中文 OCR、文档/表格解析、图表理解

### gemma-4-E4B-it（美国 / Google / Apache-2.0）
- 参数：4.5B 有效 / 8B 总（Per-Layer Embeddings 架构，"有效参数"小于"总参数"）
- 视觉塔：~150M，**16 层 / 768 隐藏维 / 12 头**（已用 `Gemma4VisionConfig` 实测确认）
- 位置编码：学习式 2D 位置 + 多维 RoPE，保持原始宽高比
- 占位符 token：`<|image|>`；模型类 `Gemma4ForConditionalGeneration`
- 上下文 128K；还支持音频（本周不用）
- 注意：**需要 `transformers >= 5.5.0`**，本机装的是 5.14.1（独立环境 `.venv-vlm`）

### Gemma 4 全家族（选型时的对比）

| 变体 | 参数 | Vision Encoder | 上下文 | 24GB 能否跑 |
|---|---|---|---|---|
| E2B | 2.3B 有效 / 5.1B 总 | ✅ ~150M | 128K | 轻松 |
| **E4B** ← 本周选它 | 4.5B 有效 / 8B 总 | ✅ ~150M | 128K | bf16 推理 ~16GB，LoRA ~17GB |
| 12B Unified | 11.95B | ❌ **encoder-free** | 256K | 需 4-bit |
| 26B-A4B (MoE) | 25.2B / 3.8B 激活 | ✅ ~550M | 256K | LoRA 需 >40GB ✗ |
| 31B Dense | 30.7B | ✅ ~550M | 256K | QLoRA 22GB，太顶 |

选 E4B 的三条理由：① 有真 ViT（12B Unified 没有）；② 24GB 上推理和 LoRA 都跑得动；
③ Apache-2.0 无门禁（对比 Llama-3.2-Vision 在 HF 上是 gated 且 EU 禁用多模态）。

## 五、为什么不按任务书选 2B

任务书按 8GB 显存写"务必选 2B 级 VLM"。实测本机 **RTX 4090 24GB**，条件不成立。
更重要的是实验有效性：**2B 级模型的 OCR 幻觉率本身就很高**，
Day25 要量化幻觉率，如果基座本身就在乱猜，测出来的是"小模型能力不足"而不是"幻觉行为"，
结论没有分析价值。7B/8B 级是能得出有意义结论的最低档。

## 六、Day22 执行记录

1. 建 `Week5/{code,configs,data,deliverables,docs}`
2. 建独立环境 `.venv-vlm`：torch 2.6.0+cu124 / transformers 5.14.1 / qwen-vl-utils 0.0.14 /
   bitsandbytes 0.50.0 / peft 0.20.0（脚本 `setup_venv_vlm.ps1`）
3. ModelScope 下载两模型（`download_vlm.py`，失败自动回退 hf-mirror.com）
4. 素材包：`prepare_images.py` 自动生成 3 张真值 100% 可控的图
   （表格截图用 Week4 真实 DPO 数据渲染 / UI 界面合成 / 业务图表复用 Week4 产出），
   另外 3 张（风景、Logo、手写公式）必须是真实照片，人工放入 —— 见 `data/README_图片素材.md`
5. `ground_truth.json` 记录每张图的 `present_objects` / `absent_objects` / `exact_text`，
   **这是 Day25 判定物体幻觉的唯一依据**，必须在跑推理之前定好

## 七、下载踩坑（花了最多时间的地方）

`Qwen2.5-VL-7B` 的 5 个分片里，shard 1 在 ModelScope 上走到了一个慢节点：

| 现象 | 数据 |
|---|---|
| shard 2/3/4/5 从 ModelScope 正常下完 | 聚合约 6.5 MB/s |
| shard 1 单连接极慢 | 0.6 ~ 1.4 MB/s |
| 中途一次 `Hash validation failed, retrying (1/3)` | 已下的 3.0 GB 全部作废重来 |
| **同一时刻同一条网络，hf-mirror.com 实测** | **17 MB/s（12×）** |

三条经验：

1. **不要盲等**。发现某个分片速度掉到 1 MB/s 量级，先测一下另一个源再决定，
   25 秒的探测能省 40 分钟。
2. **多流比单流快得多**。Gemma（单个 16 GB 文件）从 ModelScope 拿到 11.5 MB/s，
   而 Qwen 单分片只有 1.4 MB/s —— 瓶颈在单连接速率，不在总带宽。
   所以两个模型**并行下载**比串行快接近一倍。
3. **`hf_hub_download` 在 hf-mirror 上用不了**。huggingface_hub 1.26 会先请求
   Xet 元数据，镜像不返回这些头，直接抛
   `FileMetadataError: Distant resource does not seem to be on huggingface.co`。
   退到裸 `requests` 流式下载 + `Range` 续传即可（`fetch_missing_shard.py`）。

另外：ModelScope 的 `.incomplete` 断点格式和自写的 `.part` 不兼容，换源前要先删掉残留，
否则会把两段不同来源的字节拼在一起，最后 safetensors 反序列化报错，很难定位。

## 八、踩坑记录（其他）

- **PowerShell 向 python.exe 传 `-c` 脚本时会吞掉双引号**，脚本里的字符串字面量会被破坏。
  临时探测代码要写成 `.py` 文件再执行，不要用 `python -c`。
- **Windows 上 `.incomplete` 下载临时文件的大小不实时刷新**（NTFS 目录项在句柄关闭前不更新），
  用 `Get-ChildItem` 测下载进度会看到 0 MB/s 的假象，要看下载器自己的日志。
- **Consolas 没有中文字形**，用它渲染含中文的字符串会出现方块（豆腐块）。
  生成测试图时要按字符串是否含中日韩字符切换字体。
