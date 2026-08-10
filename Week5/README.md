# 第 5 周：多模态实践（VLM）

> 环境：Windows 11 + RTX 4090 (24GB)，**独立环境 `.venv-vlm/`**（Python 3.12 + transformers 5.14.1）。
> 从纯文本模型切换到视觉语言模型，核心目标是理解**图像如何被"翻译"成文本空间**。
> 本周做 **选型下载 → 图文推理与能力边界 → 跨模态注意力可视化 → 幻觉量化 → VLM LoRA 微调 → 周报**。

## 一、选型（与任务书的偏差及理由）

任务书按 8GB 显存建议 Qwen2-VL-2B。**实测本机为 RTX 4090 24GB，因此上调到 7B/8B 级**：

| 模型 | 归属 | 参数 | 视觉编码器 | 视觉 token 策略 | 许可 |
|---|---|---|---|---|---|
| **Qwen2.5-VL-7B-Instruct** | 中国 / 阿里 | 7B + 675M ViT | 原生动态分辨率 ViT | **动态**，≈(H/28)×(W/28)，2×2 merge | Apache-2.0 |
| **gemma-4-E4B-it** | 美国 / Google | 4.5B 有效 / 8B 总 | ~150M ViT（16层/768维/12头） | **固定预算** 70/140/280/560/1120 | Apache-2.0 |

- 不选 Qwen2-VL-2B：2B 的 OCR 幻觉过重，会污染 Day25 的幻觉率结论。
- 美国对照模型选 Gemma 4 而非 Llama-3.2-Vision：Gemma 4 **Apache-2.0 且无门禁**（Llama 3.2 Vision 在 HF 上是 gated 且 EU 禁用多模态）。
- Gemma 4 五个变体里，**E2B / E4B 有 ~150M ViT，26B-A4B / 31B 有 ~550M ViT，
  而 12B Unified 是 encoder-free（无 ViT，raw patch 直接线性投影）**。
  需要真 ViT + 24GB 要能跑 LoRA（E4B 约 17GB），故选 **E4B**。
- **两者是同一范式的两个极端**：都用 soft-token 注入 + 纯 self-attention（无 cross-attention），
  但一个 token 数随图动态变化、一个固定预算。这条轴贯穿 Day23/24/25 的全部对比。

### 为什么必须建独立环境 `.venv-vlm`

Gemma 4 需要 `transformers >= 5.5.0`，主环境 `.venv` 是 4.56.2（Week1–4 的脚本依赖它）。
直接升级会打断前四周的可复现性，因此隔离。Qwen2.5-VL 在 transformers 5.x 下同样可用，
所以 Week5 两个模型共用 `.venv-vlm`（实装 transformers **5.14.1**）。

### LLaMA-Factory 与 transformers 5.14.1 的共存方案（已实测跑通）

本地 LF 0.9.6.dev0（@2026-07-08）把 transformers 钉在 `>=4.55.0,<=5.7.0`。
直接 `pip install -e ./LLaMA-Factory` 会把 transformers 降级、反过来搞坏 Gemma-4。
`setup_venv_vlm.ps1` 里已固化的解法：

| 步骤 | 做法 | 不这么做会怎样 |
|---|---|---|
| ① | `pip install -e ./LLaMA-Factory **--no-deps**` | pip 会连带降级 transformers / accelerate / datasets / peft |
| ② | 手工补齐缺失依赖，**跳过**上述四个 | LF 缺 `omegaconf`/`fire`/`tyro` 等直接 ImportError |
| ③ | `trl` 钉 **0.24.0**（`--no-deps`） | trl 1.x 删了 `AutoModelForCausalLMWithValueHead`，而 LF 的 `model/loader.py` 在模块加载时就 import 它 |
| ④ | `torchaudio` 钉 **2.6.0** 匹配 torch 2.6.0 | 版本不匹配时 `_torchaudio.pyd` 加载失败，报 `WinError 127` |
| ⑤ | 运行时 `DISABLE_VERSION_CHECK=1` | LF 的 `check_dependencies()` 会硬断言 transformers ≤5.7.0 并抛 ImportError |

做完这五步，`llamafactory.train.tuner` / `TEMPLATES` / `COMPOSITE_MODELS` 全部可正常导入，
`qwen2_vl` 与 `gemma4` 模板、`qwen2_5_vl` 与 `gemma4` composite model 均已确认注册。
（若后续训练时仍撞上 5.x 的 API 变更，退路是把 transformers 钉回 5.7.0——
Gemma-4 支持从 5.5.0 起，仍然可用。）

## 二、目录结构

```
Week5/
├── code/
│   ├── setup_venv_vlm.ps1      Day22 创建 .venv-vlm（transformers>=5.5 + qwen-vl-utils + bnb）
│   ├── download_vlm.py         Day22 ModelScope 下载两模型（失败自动回退 hf-mirror）
│   ├── fetch_missing_shard.py  Day22 补下缺失分片：直连 hf-mirror + 断点续传（见下方踩坑）
│   ├── prepare_images.py       Day22 生成表格/UI/图表三张图 + ground_truth.json
│   ├── validate_data.py        跑模型前的数据自检（36 项：图片存在性/真值完整性/正负平衡/LF 格式）
│   ├── vlm_common.py           全周共用：加载/消息构造/生成/参数量拆解（保证对比公平）
│   ├── check_models.py         Day22 交付：下载确认 + 参数量拆解 + 显存/延迟/视觉token 实测
│   ├── build_questions.py      Day23 生成 5类×6图 问题矩阵
│   ├── vlm_infer.py            Day23 批量图文推理 → 结果表 CSV（支持 --resume）
│   ├── attn_hook.py            Day24 hook 抽 self-attn 的 text→image 子块 → npz
│   ├── plot_attn.py            Day24 热力图叠加 / 逐层演化图 / 图像注意力占比
│   ├── build_hallu_probes.py   Day25 三类幻觉探针（存在性/误导前提/迎合诱导）
│   ├── hallu_eval.py           Day25 跑探针 + 指标统计 + 报告（支持 --tally）
│   ├── build_vlm_sft_data.py   Day26 生成 200 训练 + 20 留出（图与真值同时生成）
│   ├── compare_finetune.py     Day26 微调前后客观对比（字段级准确率）
│   ├── run_day26.ps1           Day26 重跑流水线：LoRA 训练 → 挂 adapter 对比
│   └── md_to_docx.py           Day27 周报转 docx（原样复用 Week4）
├── configs/
│   ├── qwen2_5vl_lora_sft.yaml    Day26 主训练配置（冻结 ViT + LoRA r16）
│   ├── gemma4_e4b_lora_sft.yaml   Day26 对照组，超参与主配置完全一致
│   └── merge_qwen2_5vl_lora.yaml  Day26 adapter 合并导出
├── data/
│   ├── images/                 6 张测试图 + ground_truth.json（真值已全部填妥）
│   ├── README_图片素材.md       6 张图的来源、尺寸、考察点
│   ├── questions.json          Day23 问题矩阵（核心 25 条 + 加分 5 条）
│   ├── hallucination_probes.json  Day25 探针 53 条（存在性正负 17:17）
│   ├── attn_npz/               Day24 抓取的注意力权重
│   ├── train_images/ (200)     Day26 训练图，随机化超参/版式/配色
│   ├── eval_images/  (20)      Day26 留出图，独立随机流，与训练集不重叠
│   ├── vlm_sft_200.json        Day26 训练集（ShareGPT 格式 + images 字段）
│   ├── eval_records.json       Day26 留出集真值（8 字段/条）
│   └── dataset_info.json       LLaMA-Factory 数据集注册（week5_vlm_sft）
├── docs/                       Day22~27 每日工作说明
└── deliverables/               交付物（见下）
```

## 三、运行顺序（在 SenceTime_Week1/ 根目录）

```powershell
# ① Day22 环境 + 下载 + 素材（下载约 1.5 小时，33GB）
powershell -ExecutionPolicy Bypass -File Week5\code\setup_venv_vlm.ps1
.\.venv\Scripts\python.exe Week5\code\download_vlm.py          # 只依赖 modelscope，用主环境即可
#   若某个分片卡在 <2 MB/s 或报 Hash validation failed，换 hf-mirror 单独补这一片：
.\.venv-vlm\Scripts\python.exe Week5\code\fetch_missing_shard.py Qwen2.5-VL-7B-Instruct
.\.venv\Scripts\python.exe Week5\code\prepare_images.py        # 生成 01/05/06 + ground_truth
.\.venv-vlm\Scripts\python.exe Week5\code\check_models.py      # → Day22_模型下载确认.md
.\.venv-vlm\Scripts\python.exe Week5\code\build_questions.py
.\.venv-vlm\Scripts\python.exe Week5\code\build_hallu_probes.py
.\.venv-vlm\Scripts\python.exe Week5\code\build_vlm_sft_data.py
.\.venv-vlm\Scripts\python.exe Week5\code\validate_data.py     # 36 项自检，跑模型前必过

# ② Day23 批量图文推理（两模型 × 30 问，缺图的问题自动跳过；补图后 --resume 续跑）
.\.venv-vlm\Scripts\python.exe Week5\code\vlm_infer.py
#   → deliverables\图文推理结果表.csv（score / hallucination / note 三列需人工填）

# ③ Day24 注意力可视化（注意 attn_hook 内部强制 eager，不要改成 sdpa）
.\.venv-vlm\Scripts\python.exe Week5\code\attn_hook.py --model qwen --image 01_table.png `
    --layers 5 14 20 27 --question "这张表里 eval acc 最高的是哪一行？它的峰值显存是多少？"
.\.venv-vlm\Scripts\python.exe Week5\code\plot_attn.py --npz Week5\data\attn_npz\qwen_01_table.npz --list-tokens
.\.venv-vlm\Scripts\python.exe Week5\code\plot_attn.py --npz Week5\data\attn_npz\qwen_01_table.npz `
    --anchor 13.5 --layer 20 --evolution
#   同一张图对 gemma 再跑一遍 → 两种视觉 token 策略的注意力对比

# ④ Day25 幻觉检测
.\.venv-vlm\Scripts\python.exe Week5\code\build_hallu_probes.py
.\.venv-vlm\Scripts\python.exe Week5\code\hallu_eval.py
#   ↑ 复核 deliverables\幻觉检测明细.csv 里 judge=unclear 的行，填「人工复核」列后：
.\.venv-vlm\Scripts\python.exe Week5\code\hallu_eval.py --tally

# ⑤ Day26 微调（先出基线，再训练，再对比）
.\.venv-vlm\Scripts\python.exe Week5\code\build_vlm_sft_data.py
.\.venv-vlm\Scripts\python.exe Week5\code\compare_finetune.py --tag base
.\.venv-vlm\Scripts\python.exe -m llamafactory.cli train Week5\configs\qwen2_5vl_lora_sft.yaml
.\.venv-vlm\Scripts\python.exe Week5\code\compare_finetune.py --tag lora --adapter saves\qwen2.5vl-7b-week5-lora
.\.venv-vlm\Scripts\python.exe -m llamafactory.cli export Week5\configs\merge_qwen2_5vl_lora.yaml

# ⑥ Day27 周报转 docx
.\.venv-vlm\Scripts\python.exe Week5\code\md_to_docx.py Week5\deliverables\第5周_多模态实践报告.md
```

> **Day26 跑训练前必须先设环境变量**：
> ```powershell
> $env:DISABLE_VERSION_CHECK = "1"
> ```
> 原因见下面《LLaMA-Factory 与 transformers 5.14.1 的共存方案》。
>
> **⚠ 不要设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。**
> 实测（torch 2.6.0 + Windows）开启后 allocator 一上来就占满 23.7/24 GB，
> 步速从 5.4 s/step 劣化到 23→31→38 s/step 且持续恶化。
> 这个参数在 Linux 上是抗碎片标准手段，**不要跨平台照搬**。
> 第 ⑤ 步的训练+对比已经串成 `run_day26.ps1`，直接跑它即可。

## 四、每日任务与交付物

| Day | 任务 | 交付物 | 关键坑 |
|---|---|---|---|
| 22 | 选型、下载、素材包 | `Day22_模型下载确认.md`、`data/images/` + `ground_truth.json` | Gemma4 要 transformers≥5.5，必须隔离环境 |
| 23 | 5类问题 × 5图 图文推理 | `图文推理结果表.csv`（两模型共 50 行）、`Day23_能力边界分析.md` | `max_pixels` 不设必 OOM |
| 24 | 跨模态注意力可视化 | `attn/*.png`（≥4 张）、`Day24_注意力可视化分析.md` | **必须 `attn_implementation="eager"`**，SDPA/FlashAttn 不返回注意力矩阵 |
| 25 | 幻觉检测与量化 | `幻觉检测报告.md` + 明细 CSV + 对比图 | 存在性探针**正负样本各半**，否则一律答"是"就能刷高分 |
| 26 | VLM LoRA 微调（冻结 ViT） | 微调模型 + `logs/` + `微调前后对比表.md` | `freeze_vision_tower: true`；`image_max_pixels` 是训练显存命门 |
| 27 | 补漏 + 周报 | `第5周_多模态实践报告.md` / `.docx` | —— |

## 五、三个贯穿全周的技术要点

1. **图像进入 LLM 的三步**：ViT 切 patch 编码 → projector/merger 映射到 LLM 的 hidden_size →
   这些向量占据 `<|image_pad|>`（Qwen）/ `<|image|>`（Gemma）占位符的位置，之后与文本 token 一视同仁。
   **一张图 = 几百到几千个 token**，这是显存与延迟的第一解释变量。

2. **本周两个模型都没有 cross-attention**。任务书 24.1 写的"提取 Cross-Attention 权重"在
   Qwen2.5-VL 和 Gemma 4 上都不存在——它们是 soft-token 注入 + 纯 self-attention。
   正确做法是取 self-attention 矩阵里 `text_token → image_token` 的**子块**。
   （真有 cross-attn 层的是 Llama-3.2-Vision，本周作为架构对比在文档里讨论，不下载。）

3. **冻结 ViT 的依据**：视觉编码器只占总参数的很小一部分（见 `Day22_模型下载确认.md` §三），
   且它是在几十亿图文对上预训练的通用特征提取器。200 条数据去动它只会破坏表征。
   要教的是"看到这类图该用什么语气/格式说话"——那是 LLM 侧的事。

## 六、验收标准对照

| # | 验收 | 状态 | 证据 |
|---|---|---|---|
| ❶ | 能完成图文推理 | ✅ | 60 条记录（要求 25）；关键事实命中率 Qwen 100% / Gemma 87%，`图文推理结果表.csv` |
| ❷ | 注意力可视化成功 | ✅ | 6 张热力图（要求 ≥3）；数字 token 图像注意力占比 0.274 vs 标点 0.160，`deliverables/attn/` |
| ❸ | VLM 微调后有明显效果提升 | ✅ | 字段准确率 77.7%→**98.2%**，整条全对率 0%→**80%**，派生型字段 40%→**93.3%**，`微调前后对比表.md` |
| ❹ | 周报提交 | ✅ | `第5周_多模态实践报告.md` / `.docx` + `docs/Day22~27` |
