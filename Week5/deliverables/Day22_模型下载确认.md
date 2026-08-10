# Day22 交付：VLM 模型下载确认

> 生成时间：2026-08-05 13:14　由 `Week5/code/check_models.py` 自动生成，手改会被覆盖。

## 一、硬件与环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090　24.0 GB |
| torch / CUDA | 2.6.0+cu124 / 12.4 |
| transformers | 5.14.1 |
| Python / OS | 3.12.10 / Windows 11 |
| 虚拟环境 | `.venv-vlm`（独立环境，因 Gemma-4 需要 transformers>=5.5，与主环境 4.56.2 冲突） |

### 选型说明

任务书建议「8GB 显存选 2B 级 VLM」，但实测本机为 **RTX 4090 24GB**，因此上调到 7B/8B 级：

- **Qwen2.5-VL-7B-Instruct**（而非 Qwen2-VL-2B）：2B 的 OCR 幻觉过重，会污染 Day25 的幻觉率结论；7B 在 24GB 上 bf16 可直接跑。

- **gemma-4-E4B-it**（美国对照模型）：Gemma 4 五个变体中，E2B/E4B 带 ~150M vision encoder，26B-A4B/31B 带 ~550M，而 **12B Unified 是 encoder-free（无 ViT）**。需要真 ViT + 24GB 能跑 LoRA（~17GB），故选 E4B。Apache-2.0，无门禁。

## 二、下载确认

| 模型 | 来源 | 本地体积 | 模型类 | dtype | 加载耗时 | 权重显存 |
|---|---|---|---|---|---|---|
| `Qwen2.5-VL-7B-Instruct` | 中国 / 阿里 | 15.46 GB | `Qwen2_5_VLForConditionalGeneration` | bfloat16 | 7.4s | 15.45 GB |
| `gemma-4-E4B-it` | 美国 / Google | 14.92 GB | `Gemma4ForConditionalGeneration` | bfloat16 | 8.4s | 14.8 GB |

- `Qwen2.5-VL-7B-Instruct` → `C:\Users\Ruibo's Desktop\SenceTime_Week1\models\Qwen2.5-VL-7B-Instruct`
- `gemma-4-E4B-it` → `C:\Users\Ruibo's Desktop\SenceTime_Week1\models\gemma-4-E4B-it`

## 三、参数量拆解（图像如何被翻译进文本空间）

| 模型 | 视觉编码器 | 跨模态投影 | 语言模型 | 其他 | 合计 |
|---|---|---|---|---|---|
| `Qwen2.5-VL-7B-Instruct` | 632.0 M | 44.6 M | 7,615.6 M | — | 8,292.2 M |
| `gemma-4-E4B-it` | 169.3 M | — | 7,463.0 M | 308.8 M | 7,941.1 M |

**读法**：视觉编码器只占总参数的很小一部分，说明 VLM 的「看图能力」主要不是靠视觉塔的容量，而是靠投影层把视觉特征对齐到 LLM 的 embedding 空间——这也是 Day26 微调时可以放心冻结 ViT 的直接依据。

## 四、单图推理实测（视觉 token 数 / 延迟 / 显存）

### Qwen2.5-VL-7B-Instruct

| 图片 | 视觉 token | 提示 token | 生成 token | 耗时 | 峰值显存 |
|---|---|---|---|---|---|
| `01_table.png` | 532 | 560 | 21 | 2.12s | 15.62 GB |
| `02_landscape.jpg` | 1247 | 1275 | 19 | 2.39s | 16.06 GB |
| `03_logo.jpg` | 1218 | 1246 | 26 | 2.67s | 16.03 GB |
| `04_signboard_jp.jpg` | 1271 | 1299 | 29 | 1.57s | 16.07 GB |
| `05_ui.png` | 1092 | 1120 | 24 | 1.22s | 15.94 GB |
| `06_chart.png` | 819 | 847 | 23 | 1.02s | 15.76 GB |

- **01_table.png** → 这张图片展示了Week4 DPO实验中不同配置下的末次评估与训练成本结果。
- **02_landscape.jpg** → 两个人在海滩上散步，背景是宁静的海面和远处的山丘。
- **03_logo.jpg** → 这张图片展示了纳斯达克的标志和背景中的城市景观，突出了EVgo在纳斯达克上市的信息。
- **04_signboard_jp.jpg** → 这张图片展示了一块写有日文的白色告示牌，背景是郁郁葱葱的树木和灌木丛。
- **05_ui.png** → 一张显示LoRA训练任务状态的界面，包括超参数配置、运行状态和训练日志信息。
- **06_chart.png** → 这张图片展示了三组DPO模型在不同参数设置下的margin值随训练步数的变化情况。

### gemma-4-E4B-it

| 图片 | 视觉 token | 提示 token | 生成 token | 耗时 | 峰值显存 |
|---|---|---|---|---|---|
| `01_table.png` | 546 | 565 | 72 | 4.48s | 14.97 GB |
| `02_landscape.jpg` | 532 | 551 | 29 | 1.86s | 14.97 GB |
| `03_logo.jpg` | 532 | 551 | 37 | 2.53s | 14.97 GB |
| `04_signboard_jp.jpg` | 540 | 559 | 29 | 2.09s | 14.97 GB |
| `05_ui.png` | 540 | 559 | 74 | 4.83s | 14.97 GB |
| `06_chart.png` | 527 | 546 | 61 | 5.66s | 14.97 GB |

- **01_table.png** → 这张图片展示了“Week4 DPO 实验：本次评估与训练成本”的实验结果表格，记录了不同配置（如 `qwen_dpo_beta` 和学习率）下的评估准确率（eval acc）、评估边距（eval margin）、训练损失（train loss）、耗时和存储大小等指标。
- **02_landscape.jpg** → 在青山环绕的海边沙滩上，一对情侣在海边散步，前景停放着几艘船只。
- **03_logo.jpg** → 这张图片展示了在城市背景下，一个巨大的广告牌上印有“Nasdaq”和“EVgo”的标志，突显了这两家公司的品牌形象。
- **04_signboard_jp.jpg** → 这张图片展示了一个户外信息牌，牌子上用日文写着一些文字，背景是茂密的树林和草地。
- **05_ui.png** → 这张图片展示了一个名为“LoRA 训练台”的界面，显示了一个名为“job-20260805-vlm-sft”的训练任务的详细信息，包括**超参数配置**、**运行状态**（如已完成步数、当前损失、剩余时间）以及**训练日志**。
- **06_chart.png** → 这张图展示了三种不同超参数设置的 DPO 模型在训练过程中奖励（rewards/margins）随训练步数的变化趋势，其中绿色线（qwen\_dpo\_beta0.5\_lr5e-6）的奖励值最高，表明其性能最优。

## 五、视觉 token 策略对比（本周核心认知）

- **Qwen2.5-VL**：原生动态分辨率，视觉 token 数 ≈ (H/28)×(W/28)，随图片尺寸变化；本次 `max_pixels=1,003,520`（≈100 万像素）做了上限约束。不设这个参数，一张 4K 截图的视觉 token 会上万，直接吃掉 8GB+ 显存。

- **Gemma-4-E4B**：保持原始宽高比，但 soft token 预算**固定可选** 70 / 140 / 280 / 560 / 1120（默认 280，本次用 560）。图再大 token 数也不涨，代价是细节丢失。

- 两者都是 **soft-token 注入 + 纯 self-attention** 范式（无 cross-attention），这一点决定了 Day24 的注意力可视化必须取 self-attention 矩阵中 `text_token → image_token` 的子块，而不是去找 cross-attn 层。
