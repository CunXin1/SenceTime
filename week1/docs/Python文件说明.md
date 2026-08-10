# 本周要写的 Python 文件说明(干什么 + 怎么写)

> 这周需要动手的代码产物一共 **4 个**(2 个已写好,2 个待写)。
> Day 5 的 LLaMA-Factory **不用写 .py**——用它的 `llamafactory-cli` 命令 + 一个 YAML 配置,且在 4090 上做。

## 总览

| 文件 | 对应 | 干什么 | 状态 |
|---|---|---|---|
| `download_models.py` | Day 2 | 从 HuggingFace 下载模型到本地 `models/` | ✅ 已写 |
| `inference.py` | Day 2 | 加载模型跑对话,理解推理 4 步管线 | ✅ 已写 |
| `count_params.py` | Day 3 | 统计各层参数量、验证总参数量 | ⏳ 待写 |
| `tokenizer_experiments.ipynb` | Day 4 | 分词极端用例实验(Jupyter Notebook) | ⏳ 待写 |

---

## ① `download_models.py`(已写)—— 下载模型

**干什么**:把 HF 仓库的权重 + config + tokenizer 下到项目 `models/` 下。

**核心 API 就一个**:
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen2.5-1.5B-Instruct", local_dir="./models/xxx")
```

**写法要点**:
- 用字典管理"仓库 id → 本地名 → 是否 gated"。
- gated(受限)模型先查 `get_token()` 是否已登录,没登录就跳过并提示。
- `allow_patterns=["*.safetensors","*.json",...]` 只下需要的文件,省流量。

---

## ② `inference.py`(已写)—— 原生推理

**干什么**:加载模型,把"文字 → 回答"的 4 步管线显式写出来。

```
① Tokenize  文字→token id  →  ② apply_chat_template 套 ChatML
        →  ③ generate 自回归逐token生成  →  ④ decode token id→文字
```

**核心 API**:
```python
tokenizer = AutoTokenizer.from_pretrained(path)          # 加载分词器
model = AutoModelForCausalLM.from_pretrained(path, ...)   # 加载模型
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)  # ②
inputs = tokenizer([text], return_tensors="pt").to(device)   # ①
out = model.generate(**inputs, max_new_tokens=512)           # ③
ans = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)  # ④
```

**要理解**:
- `generate` 内部是"自回归循环":预测一个 token → 拼回输入 → 再预测下一个,直到 `<|im_end|>` 或达上限。
- `apply_chat_template` 把消息包成 `<|im_start|>role...<|im_end|>`(ChatML),末尾补 `assistant` 提示引导模型作答。不套这个格式,效果明显变差。
- 设备无关:Mac 用 `mps`,4090 用 `cuda`,只改一个字符串。

---

## ③ `count_params.py`(Day 3 待写)—— 参数统计

**干什么**:遍历模型每层权重张量,统计参数量,分类汇总(embedding / attention / mlp / norm),算总数,验证"1.5B"。理解"参数堆在哪"。

**核心 API 就一个**:张量的 `.numel()`(元素个数 = 参数个数)。

**完整骨架(照着能跑)**:
```python
from transformers import AutoModelForCausalLM
from collections import defaultdict
import torch

model = AutoModelForCausalLM.from_pretrained(
    "./models/Qwen2.5-1.5B-Instruct", dtype=torch.float16)

# 1) 总参数量
total = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total:,}  ({total/1e9:.2f}B)")

# 2) 分类统计:named_parameters() 给出 (名字, 张量)
buckets = defaultdict(int)
for name, p in model.named_parameters():
    if   "embed"     in name: key = "embedding(词表)"
    elif "self_attn" in name: key = "attention(注意力)"
    elif "mlp"       in name: key = "mlp(前馈)"
    elif "norm"      in name: key = "norm(归一化)"
    else:                     key = "其他"
    buckets[key] += p.numel()

for k, v in sorted(buckets.items(), key=lambda x: -x[1]):
    print(f"{k:18s} {v:>13,}  {v/total*100:5.1f}%")
```

**要理解的观察点**:
- MLP 通常占最大头(SwiGLU 有 gate/up/down 三个大矩阵)。
- 小模型里 embedding(词表 ~15万 × hidden)占比意外高。
- Qwen 用 **tie embedding**(输入/输出词表共享权重),别重复计数。

---

## ④ `tokenizer_experiments.ipynb`(Day 4 待写)—— 分词实验

**干什么**:不是脚本,是 **Jupyter Notebook**(一格一格跑,方便逐个用例看结果)。设计 10+ 极端用例,观察 Qwen 的分词行为。

**核心 API 三个**:
```python
tok = AutoTokenizer.from_pretrained("./models/Qwen2.5-1.5B-Instruct")
ids  = tok.encode("你好 world")     # 文字 → id
toks = tok.tokenize("你好 world")    # 文字 → 子词(看清怎么切)
back = tok.decode(ids)              # id → 文字
```

**每个 cell 的模板(复制改文本即可)**:
```python
def show(text):
    toks = tok.tokenize(text)
    ids  = tok.encode(text)
    print(f"文本: {text!r}")
    print(f"token数: {len(ids)}")
    print(f"切分: {toks}")
    print(f"ids : {ids}\n")

show("你好世界")              # 纯中文:一个字几个token
show("Hello world")          # 纯英文:一个词几个token
show("你好world混合123")      # 中英数混合
show("😀🎉 emoji 表情")       # emoji(byte级BPE拆多字节)
show("∫∑√ x²+y²=z²")         # 数学符号
show("   连续   空格")        # 空格处理
show("<|im_start|>")         # 特殊token:是否为单个整token
show("这是一段很长的文本" * 50) # 长文本
```

**要理解的观察点(Day 4 精髓)**:
- 中文"你好世界"可能 2–4 个 token,英文"Hello"常 1 个 → **中文更费 token = 更贵**。
- `<|im_start|>` / `<|endoftext|>` 是**单个特殊 token**(id 很大),不会被拆开。
- emoji / 数学符号走 **byte-level BPE**,一个符号可能拆成几个字节 token。
- 对比 BPE(Qwen)vs SentencePiece(Llama/Gemma)在中文上的切分差异。

---

## 与交付物的对应

| Python 产物 | 交付物 |
|---|---|
| `inference.py` + 运行日志 | Day 2:推理脚本 + 3 组对话日志 |
| `count_params.py` + 输出 | Day 3:参数统计脚本(附架构报告) |
| `tokenizer_experiments.ipynb` | Day 4:Tokenizer 实验 Notebook(10+ 用例) |
