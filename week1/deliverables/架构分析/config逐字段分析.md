# Qwen2.5-1.5B config.json 逐字段分析 (Day 3.1)

> 逐个解释 `models/Qwen2.5-1.5B-Instruct/config.json` 的**每一个字段**,重点标注任务要求的
> vocab_size / hidden_size / num_attention_heads / num_key_value_heads / rope_theta。

## 完整 config.json

```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 1536,
  "initializer_range": 0.02,
  "intermediate_size": 8960,
  "max_position_embeddings": 32768,
  "max_window_layers": 21,
  "model_type": "qwen2",
  "num_attention_heads": 12,
  "num_hidden_layers": 28,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.43.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
```

---

## 一、任务重点字段(5 个)

### ⭐ `vocab_size: 151936` —— 词表大小
- **含义**:模型认识 151936 个不同的 token。
- **推导**:embedding 矩阵 = vocab × hidden = 151936 × 1536 ≈ **2.33 亿参数**(占 1.5B 的 15.1%)。
- **意义**:词表很大(Llama3 才 128256),对中文/多语言友好,同样中文更省 token。

### ⭐ `hidden_size: 1536` —— 隐藏维度(模型"宽度")
- **含义**:每个 token 用 1536 维向量表示,是贯穿全模型的主干宽度。
- **推导**:`head_dim = hidden_size / num_attention_heads = 1536 / 12 = 128`(每个注意力头 128 维)。
- **意义**:宽度决定表达容量;1536 是 1.5B 这个体量的典型选择。

### ⭐ `num_attention_heads: 12` —— Query 头数
- **含义**:注意力有 12 个并行的 Query 头,每头 128 维,并行捕捉不同关系。

### ⭐ `num_key_value_heads: 2` —— KV 头数(GQA)
- **含义**:只有 2 个 Key/Value 头,被 12 个 Q 头共享 → **GQA,分组比 6:1**。
- **意义**:推理时 KV cache 只有全 MHA 的 **1/6**,大幅省显存、加速长文本。这是 Qwen2.5 attention 只占 10% 参数的直接原因。

### ⭐ `rope_theta: 1000000.0` —— RoPE 旋转基础频率
- **含义**:位置编码旋转的基频。越大 → 旋转越慢 → 能编码越长序列而不"绕圈"。
- **意义**:配合 `max_position_embeddings: 32768` 支撑长上下文;Llama3 才 500000,Qwen 更激进。

---

## 二、其余字段逐个解释

| 字段 | 值 | 含义 |
|---|---|---|
| `architectures` | `Qwen2ForCausalLM` | 用哪个模型类加载 —— **Qwen2.5 复用 Qwen2 架构类**(2.5 是 2 的数据/训练升级,架构同源) |
| `model_type` | `qwen2` | 模型类型标识,transformers 靠它找对应实现 |
| `num_hidden_layers` | 28 | 层数(模型"深度")。深度 × 宽度共同决定容量 |
| `intermediate_size` | 8960 | MLP 中间层维度,约 hidden 的 5.8 倍。SwiGLU 有 gate/up/down 三个矩阵,是参数大头(~75%) |
| `hidden_act` | `silu` | 激活函数;配门控即 **SwiGLU**,比 ReLU/GELU 表达力强 |
| `rms_norm_eps` | 1e-06 | **RMSNorm** 的极小值,防止除以零,保数值稳定 |
| `tie_word_embeddings` | true | 输入 embedding 与输出 lm_head **共享同一矩阵**,省下一份 2.33 亿参数(小模型关键优化) |
| `max_position_embeddings` | 32768 | 训练支持的最大上下文长度(32K token) |
| `sliding_window` | 32768 | 滑动窗口大小(限制注意力只看最近 N 个 token) |
| `use_sliding_window` | false | **未启用**滑动窗口 → 实际是全局注意力(该字段为未来预留) |
| `max_window_layers` | 21 | 若启用滑窗,前多少层用 —— 当前因 use_sliding_window=false 不生效 |
| `attention_dropout` | 0.0 | 注意力 dropout 率,推理为 0(不随机丢弃) |
| `initializer_range` | 0.02 | 权重初始化的正态分布标准差(训练时用) |
| `bos_token_id` | 151643 | 序列开始 token id(`<|endoftext|>`) |
| `eos_token_id` | 151645 | 序列结束 token id(`<|im_end|>`),生成到它就停 |
| `torch_dtype` | `bfloat16` | 权重存储精度(bf16);Mac 上我们加载为 fp16 |
| `use_cache` | true | 推理时启用 **KV cache**(缓存已算过的 K/V,逐 token 生成加速) |
| `transformers_version` | 4.43.1 | 保存该 config 时的 transformers 版本(仅记录) |

---

## 三、从 config 能推出的模型"体检表"

| 指标 | 值 | 由哪些字段推出 |
|---|---|---|
| 每头维度 head_dim | 128 | hidden_size / num_attention_heads = 1536/12 |
| GQA 分组比 | 6:1 | num_attention_heads / num_key_value_heads = 12/2 |
| KV cache 压缩 | 6 倍 | 同上 |
| embedding 参数 | ~2.33 亿 | vocab × hidden = 151936×1536 |
| 是否长文本友好 | 是(32K) | max_position_embeddings + 大 rope_theta |
| 归一化方式 | RMSNorm(Pre-Norm) | rms_norm_eps 存在 |
| 前馈类型 | SwiGLU | hidden_act=silu + 门控结构 |

**一句话**:仅凭一份 config.json,就能读出 Qwen2.5-1.5B 是"**28 层 × 1536 宽、12/2 头激进 GQA、15 万大词表、100 万 rope_theta 长上下文、tie embedding 省参数**"的高效中文友好模型。
