# Day24：跨模态注意力可视化

## 一、动手前必须先确认：这个模型有没有 Cross-Attention

任务书 24.1 写「使用 Hooks 提取 VLM 中间层（如第 20 层）的 Cross-Attention 权重」。
**在本周的两个模型上，这个模块不存在。**

| 范式 | 图像特征去哪了 | 代表模型 | 有 cross-attn 吗 |
|---|---|---|---|
| A. soft-token 注入 | 变成 token 插进文本序列 | **Qwen2.5-VL、Gemma 4** | ❌ 只有 self-attention |
| B. cross-attention 注入 | 不进文本序列，走专用层 | Llama-3.2-Vision、Flamingo | ✅ |
| C. encoder-free | 没有 ViT，raw patch 直接投影 | Gemma 4 12B Unified、Fuyu | ❌ |

已核对 transformers 5.14.1 源码：`Qwen2_5_VLForConditionalGeneration` 和
`Gemma4ForConditionalGeneration` 的模块树里没有任何 `cross_attn`。

**正确做法**：取 self-attention 矩阵中 `text_token → image_token` 的子块

```python
attn[0, head, text_pos, img_start:img_end]     # 生成这个字时，在看图的哪里
```

语义上等价于「跨模态注意力」，但实现上寄生在 self-attention 里。
**先确认范式再动手，否则会花半天去找一个不存在的模块。**

## 二、两个必踩的实现坑

### ① 必须 `attn_implementation="eager"`

SDPA 和 FlashAttention **从不显式构造注意力矩阵**——这正是它们快且省显存的原因。
`output_attentions=True` 在它们下面返回 `None` 或直接报错。
本机装了 flash-attn 2.8.3，默认走 sdpa，不改这一项什么都拿不到。

代价：eager 的注意力是 `[B, heads, L, L]` 显式张量，L≈2000 时单层就是几百 MB。
所以 hook 里**只保留「答案位置 → 图像位置」的子块**再落盘。

### ② 不要在 `generate()` 循环里抓

带 KV cache 时每步注意力形状是 `[B, heads, 1, past+1]`，逐步拼接极易错位。
两段式做法：

```
第一段  正常 generate 得到答案（快，走 KV cache）
第二段  把 (prompt + 答案) 拼成完整序列，做一次 eager 前向（teacher forcing）
        → 等价于生成时的注意力，且事后可任选锚点 token 重新出图
```

第二段的额外好处：**出图时想换锚点不用重新推理**，`plot_attn.py --list-tokens` 列出所有
可选 token，选中哪个就画哪个。

hook 挂在 `model.model.language_model.layers[i].self_attn`，两个模型路径一致，
`forward` 都返回 `(attn_output, attn_weights)`。

## 三、一维 token 序列 → 二维网格的还原

画热力图的前提是把图像 token 摆回二维。两个模型公式不同：

- **Qwen**：`image_grid_thw` 给出 patch 网格 `(h, w)`（patch=14），merger 做 2×2 合并
  → token 网格 `(h/2, w/2)`
- **Gemma**：等比缩放到 patch 数 ≤ `max_soft_tokens × pooling²` 且边长为 48 的倍数
  （patch 16 × pooling 3）→ soft token 网格 `(H/48, W/48)`

**验证方式**：算出的 `h×w` 必须精确等于实际视觉 token 数。
实测 546 = 14×39（Gemma）、532 = 14×38（Qwen）、1247 = 29×43（Qwen），全部吻合。
对不上就说明公式错了，画出来的热力图是乱的——**这一步必须先验证再出图**。

## 四、怎么证明「模型确实在看图」

肉眼看热力图"觉得挺准"不算证据。用一个可量化的对照：

**数字 token 的图像注意力占比 vs 标点 token 的图像注意力占比**

标点由语言模型先验决定，不需要看图；数字必须从图里读。如果模型真在看图，两者应该差很多：

| 模型 | 数字 token | 标点 token | 倍数 |
|---|---|---|---|
| `Qwen2.5-VL-7B` | 0.274 | 0.160 | 6~9×（逐 token 看） |
| `gemma-4-E4B-it` | 0.283 | 0.209 | 较小 |

另加**归一化熵** = 空间注意力分布的熵 / log(N)，接近 0 表示锐利，接近 1 表示弥散。
表格图 0.762~0.900，风景图 0.889——结构化图像的注意力明显更集中。

## 五、执行

```powershell
# 抓（注意 attn_hook 内部强制 eager，不要改成 sdpa）
.\.venv-vlm\Scripts\python.exe Week5\code\attn_hook.py --model qwen --image 01_table.png `
    --layers 5 14 20 27 --question "这张表里 eval acc 最高的是哪一行？它的峰值显存是多少？"
# 列出可选锚点 token
.\.venv-vlm\Scripts\python.exe Week5\code\plot_attn.py --npz Week5\data\attn_npz\qwen_01_table.npz --list-tokens
# 出图：单层叠加 + 逐层演化
.\.venv-vlm\Scripts\python.exe Week5\code\plot_attn.py --npz Week5\data\attn_npz\qwen_01_table.npz `
    --anchor 13.5 --layer 20 --evolution
```

交付 6 张图（要求 ≥3）：qwen×表格、qwen×风景、gemma×表格，每组单层 + 逐层各一张。

## 六、结论摘要

详见 `deliverables/Day24_注意力可视化分析.md`。三条：

1. **注意力落在该看的地方**：Qwen 生成 `13.5` 时精确命中「峰值显存」列的两个单元格。
2. **浅层看纹理、深层看语义**：第 5 层弥散全图，第 14 层起收敛到目标列并保持稳定。
3. **★ Gemma 的最强激活落在表格右侧空白边缘**（attention sink），目标区域注意力预算不足，
   加上注意力头只有 8 个（Qwen 28 个）——**这正好解释了 Day23 的「数字对、形近字错」**。
   注意力可视化在这里不是画个好看的图，而是给出了 OCR 精度差异的机理。
