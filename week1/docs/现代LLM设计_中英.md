# 现代 LLM 架构设计详解 / Modern LLM Design Explained (中英)

> 覆盖 RoPE、GQA 及其它当代主流设计,配合本项目 4 个模型(Qwen2.5 / Qwen3 / Llama3.2 / Gemma4)的真实 config 数值。
> Covers RoPE, GQA and other mainstream designs, with real config values from the 4 models in this project.

---

## 1. RoPE — 旋转位置编码 / Rotary Position Embedding

**中文**
注意力机制本身没有顺序感("猫追狗"与"狗追猫"在它眼中一样),必须注入位置信息。RoPE 的做法是:**按 token 的位置,把它的 Query、Key 向量旋转一个角度**,位置越靠后转得越多。旋转后两个向量做点积,结果**只取决于相对距离 (m−n)**,因此 RoPE 天然编码的是**相对位置**。参数 `rope_theta` 是旋转的基础频率——**越大转得越慢,可编码的位置范围越长**,更利于长文本外推。RoPE **没有可训练参数**,是即时施加在 Q/K 上的。

**English**
Attention is order-agnostic by itself, so position must be injected. RoPE **rotates each token's Query and Key vectors by an angle proportional to its position**. After rotation, the dot product between two tokens depends **only on their relative distance (m−n)**, so RoPE naturally encodes **relative position**. The `rope_theta` base frequency controls rotation speed — **larger theta = slower rotation = longer encodable range**, which helps long-context extrapolation. RoPE has **no trainable parameters**; it is applied on-the-fly to Q/K.

**真实数值 / Real values**: Qwen2.5=1,000,000 · Qwen3=1,000,000 · Llama3.2=500,000 · 初代Llama=10,000 → 越新越大 / newer = larger.

---

## 2. GQA — 分组查询注意力 / Grouped-Query Attention

**中文**
注意力有 Query / Key / Value 三套"头"。GQA 让**多个 Query 头共享少数几套 Key/Value 头**。目的是缩小推理时的 **KV cache**(每生成一个新 token 都要回看前文所有 K、V,是长文本推理的显存瓶颈)。减少 KV 头数 = KV cache 更小 = 省显存、加速,而 Q 头不变 → 质量几乎不降。它是 **MHA(全头,质量好但费显存)** 与 **MQA(单 KV,极省但掉质量)** 的折中。

**English**
Attention has Query / Key / Value heads. GQA lets **multiple Query heads share a small number of Key/Value heads**, shrinking the **KV cache** (every new token attends to all past K/V — the memory bottleneck of long-context inference). Fewer KV heads = smaller cache = less memory + faster, while keeping all Q heads preserves quality. It is the middle ground between **MHA** (all heads, best quality, heavy) and **MQA** (single KV, cheapest, quality drop).

**真实数值 / Real values (Q头:KV头 = 分组比)**:
| 模型 | Q heads | KV heads | 分组比 group ratio |
|---|---|---|---|
| Qwen2.5-1.5B | 12 | 2 | 6:1 |
| Qwen3-4B | 32 | 8 | 4:1 |
| Llama3.2-3B | 24 | 8 | 3:1 |
| Gemma4-E2B | 8 | 1 | 8:1(接近 MQA / near-MQA) |

---

## 3. SwiGLU — 门控前馈激活 / Gated Feed-Forward Activation

**中文**
MLP(前馈层)是参数最密集的部分(占 65–75%)。现代模型用 **SwiGLU**:用两条并行线性层,一条经 SiLU 激活当"门",逐元素相乘再投影回去。比传统 ReLU/GELU 表达力更强。config 里 `hidden_act: silu` + 三个矩阵(gate/up/down)就是它。

**English**
The MLP is the most parameter-dense block (65–75%). Modern models use **SwiGLU**: two parallel linear projections, one passed through SiLU as a "gate", multiplied element-wise, then projected back. More expressive than plain ReLU/GELU. In config, `hidden_act: silu` with three matrices (gate/up/down) indicates SwiGLU.

---

## 4. RMSNorm — 均方根归一化 / Root-Mean-Square Normalization

**中文**
归一化稳定训练。**RMSNorm** 是 LayerNorm 的简化版:只按均方根缩放,不减均值、无偏置,更快且效果相当。现代模型普遍用 **Pre-Norm**(归一化放在子层之前),训练更稳定。参数极少(本项目 4 个模型 norm 都占 0.0%)。

**English**
Normalization stabilizes training. **RMSNorm** is a simplified LayerNorm: scales by root-mean-square only, without mean subtraction or bias — faster with comparable quality. Modern models use **Pre-Norm** (normalize before each sub-layer) for training stability. Very few parameters (norm = 0.0% in all 4 models here).

---

## 5. Tie Word Embeddings — 词表权重共享 / Weight Tying

**中文**
输入 embedding(词→向量)和输出 lm_head(向量→词)**共享同一套权重矩阵**,省下一大块参数。config 里 `tie_word_embeddings: True`。统计参数量时**不能重复计数**(本项目 4 个模型都为 True)。

**English**
The input embedding (token→vector) and output lm_head (vector→token) **share the same weight matrix**, saving a large chunk of parameters. `tie_word_embeddings: True` in config. When counting params, **don't double-count** (all 4 models here are True).

---

## 6. 词表大小 / Vocabulary Size

**中文**
词表越大,单个中文字/词越可能是 1 个 token(更省 token = 更省钱/更快),但 embedding 矩阵也越大。小模型里词表 embedding 占比可能很高。

**English**
A larger vocabulary means a Chinese character/word is more likely a single token (fewer tokens = cheaper/faster), but the embedding matrix grows. In small models the embedding can dominate parameters.

**真实数值 / Real values**: Gemma4=262,144 · Qwen2.5/Qwen3=151,936 · Llama3.2=128,256. → Gemma 词表最大,embedding 占其参数 54%。

---

## 7. MoE — 混合专家 / Mixture of Experts

**中文**
把一个大 MLP 拆成很多个"专家",每个 token 只激活其中几个 → **总参数很大,但每次实际计算的参数少**(省算力)。最新一代大量采用:Qwen3.6-35B-A3B(总35B/激活3B)、Llama4(Scout/Maverick)、Gemma4-26B-A4B。本项目 4 个都是 **Dense(非 MoE)**,便于入门理解。

**English**
Split one large MLP into many "experts"; each token activates only a few → **huge total parameters, but few active per step** (compute-efficient). Heavily used in the newest generation: Qwen3.6-35B-A3B (35B total / 3B active), Llama4 (Scout/Maverick), Gemma4-26B-A4B. All 4 models here are **Dense (non-MoE)** for easier learning.

---

## 8. PLE / 多模态 — Gemma 的特例 / Per-Layer Embeddings & Multimodality

**中文**
Gemma4-E2B "有效 2.3B / 原始 5.1B" 的差距来自 **Per-Layer Embeddings(PLE)**:大量 embedding 表可从加速器卸载、按需查表,不参与主计算,所以"有效"参数远小于"原始"。这解释了为什么它 embedding 占 54%。它还是**多模态**模型(文+图+音),加载需 `AutoProcessor` 而非只用 `AutoTokenizer`,聊天模板用 `<start_of_turn>...<end_of_turn>`、助手角色叫 `model`。

**English**
Gemma4-E2B's "effective 2.3B vs raw 5.1B" gap comes from **Per-Layer Embeddings (PLE)**: large embedding tables can be offloaded from the accelerator and looked up on demand, not part of the main compute — so "effective" params are far below "raw". This is why embedding is 54% of its parameters. It is also **multimodal** (text+image+audio), requiring `AutoProcessor` (not just `AutoTokenizer`); its chat template uses `<start_of_turn>...<end_of_turn>` with the assistant role named `model`.

---

## 9. 长上下文 / Long Context

**中文**
`max_position_embeddings` 是训练支持的最大上下文长度。配合大 `rope_theta` 或 RoPE scaling 可外推更长。Gemma4 与 Llama3.2 达 **131072(128K)**,Qwen3 为 40960,Qwen2.5 为 32768。部分模型(如 Gemma)还用**局部/全局混合注意力**(多数层只看邻近窗口,少数层看全局)进一步省算力。

**English**
`max_position_embeddings` is the max context length seen in training; large `rope_theta` or RoPE scaling extends it further. Gemma4 and Llama3.2 reach **131072 (128K)**, Qwen3 40960, Qwen2.5 32768. Some models (e.g. Gemma) also use **local/global hybrid attention** (most layers attend to a nearby window, a few attend globally) to save compute.

---

## 一句话总览 / One-line Summary

**中文**:当代 LLM 已形成共识骨架——**RoPE(相对位置)+ GQA(省 KV cache)+ SwiGLU(强前馈)+ RMSNorm(稳训练)+ tie embedding(省参数)**;各家在词表、GQA 力度、上下文长度、MoE/多模态上做差异化取舍。

**English**: Modern LLMs share a consensus backbone — **RoPE (relative position) + GQA (KV-cache saving) + SwiGLU (strong FFN) + RMSNorm (stable training) + tied embeddings (param saving)** — and differentiate via vocabulary, GQA aggressiveness, context length, and MoE/multimodality.
