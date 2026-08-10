# 第 1 周 TODO —— 环境搭建 & Qwen2.5 大模型导论(Mac 现做部分)

> 目标:不只是跑通,而是**看懂为什么这么设计**。
> 当前机器:MacBook (Apple Silicon / arm64, macOS 26.2),有 MPS、**无 CUDA**。
> 策略:Mac 上用 `Qwen2.5-1.5B-Instruct` 走通全流程;下周回家用 4090 换 `7B QLoRA` 复跑 GPU 相关部分。

---

## 图例
- [ ] 待办 / [x] 完成
- 🍎 现在 Mac 能做 ｜ 🟢 需 4090(下周)
- 每步含:**做什么 / 为什么 / 要理解到什么**

---

## 阶段 0:装 Conda(地基)🍎

- [x] **0.1 安装 Miniconda(Apple Silicon 版)** ✅ conda 26.3.2 已装,~/.zshrc 已配置
  ```bash
  curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
  bash ~/miniconda.sh -b -p ~/miniconda3
  ~/miniconda3/bin/conda init zsh
  ```
  之后**重开终端**让 conda 生效。
  - 为什么用 conda:大模型依赖版本咬得很死,conda 给隔离沙盒,装坏了删掉重来。
  - 为什么 arm64 版:装错成 x86 会走 Rosetta,慢且易冲突。
  - 要理解:conda = 环境管理 + 包管理;pip 只管 Python 包,conda 还能管 Python 本身。

---

## 阶段 1:建环境 + 工具链(Day 1 可做部分)🍎

- [x] **1.1 建 `llm_exp` 环境** ✅ Python 3.10.20
  ```bash
  conda create -n llm_exp python=3.10 -y
  conda activate llm_exp
  ```
  - 为什么 Python 3.10:PyTorch / LLaMA-Factory / vLLM 在 3.10 最稳;系统 3.14 太新会报错。
  - 要理解:activate 后提示符变 `(llm_exp)`,之后所有 pip 只进这个盒子。

- [x] **1.2 装 Mac 版 PyTorch(带 MPS)** ✅ torch 2.12.1
  ```bash
  pip install torch torchvision torchaudio
  ```
  - 为什么不带 CUDA:Mac 无 NVIDIA;Mac 上 pip 默认给 MPS 版。CUDA 版下周 4090 再装。
  - 要理解:同一份代码,Mac 用 `device="mps"`,4090 用 `device="cuda"`,只改一个字符串。

- [x] **1.3 装核心库** ✅ transformers 5.12.1 等
  ```bash
  pip install transformers accelerate tokenizers sentencepiece modelscope jupyter
  ```
  - transformers:加载模型/tokenizer 标准库(AutoModel/AutoTokenizer)。
  - accelerate:自动放设备、管显存。
  - tokenizers / sentencepiece:分词底层(Day 4 深挖)。
  - modelscope:国内下 Qwen 比 HuggingFace 快。
  - jupyter:Day 4 notebook 用。
  - 注:LLaMA-Factory / vLLM / OpenCompass 是重 GPU 工具 → **留到 4090**,现在装易依赖冲突。

- [x] **1.4 验证环境(Mac 版验收)** ✅ MPS=True, CUDA=False(预期);交付物见 deliverables/day1_env/
  ```bash
  python -c "import torch; print('MPS 可用:', torch.backends.mps.is_available()); print('CUDA 可用:', torch.cuda.is_available())"
  ```
  - 预期:`MPS 可用: True` / `CUDA 可用: False`。
  - 关键认知:任务要的 `cuda.is_available()==True` 在 Mac **必然 False**,不是装错——CUDA 是 NVIDIA 专有,Mac 等价物是 MPS。下周 4090 才会 True(验收❶)。
  - 📦 交付:本步截图 + `conda list` 截图(Day 1 的 Mac 部分)。

---

## 阶段 2:下载模型 + 原生推理(Day 2 全部)🍎

- [ ] **2.1 下载 Qwen2.5-1.5B-Instruct**
  ```bash
  modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir ./models/Qwen2.5-1.5B-Instruct
  ```
  - 为什么 1.5B:Mac 走通流程用小模型,轻快;下周 4090 换 7B,架构逻辑一致。
  - 为什么 Instruct 版:Instruct = 指令微调过,会对话;base 版只会续写文本。Day 2 做对话必须用 Instruct。
  - 要理解下载文件:`config.json`(架构,Day 3 主角)、`model.safetensors`(权重)、`tokenizer.*`(分词表)、`generation_config.json`(默认生成参数)。
  - 📦 交付:下载完成截图。

- [ ] **2.2 写 `inference.py`(理解推理 4 步管线)**
  1. Tokenize:文字 → token id
  2. apply_chat_template:包装成 `<|im_start|>...<|im_end|>` 对话格式
  3. Forward + Generate:自回归逐 token 生成
  4. Decode:token id → 文字
  - 要理解:模型本质只会"预测下一个 token",对话能力靠 chat template + 指令微调"包"出来。
  - 📦 交付:`inference.py`(带注释)。

- [ ] **2.3 三领域 Prompt 测试**
  代码生成 / 逻辑推理 / 角色扮演,各记录输出。
  - 理解目的:感受 1.5B 能力边界(代码/角色扮演尚可,复杂推理露怯)→ 为后面"为什么要微调/更大模型"埋线。
  - 📦 交付:3 组对话日志(`.md`)。

- [ ] **2.4 专测 Chat Template**
  打印 `tokenizer.apply_chat_template(..., tokenize=False)` 看原始字符串。
  - 要理解:这是 Qwen 的 **ChatML 格式**(system/user/assistant + `<|im_start|>`/`<|im_end|>`)。不套此格式效果明显变差——这是"能对话"的隐藏机关。

---

## 阶段 3:架构深度剖析(Day 3 全部)🍎 —— 理解重头戏

- [ ] **3.1 逐字段读 `config.json`**
  - `hidden_size` / `num_hidden_layers` / `intermediate_size`:模型多宽多深。
  - `num_attention_heads` vs `num_key_value_heads` → **GQA(分组查询注意力)**:Q 头多、KV 头少,多 Q 共享一组 KV。为什么:省显存、加速推理(KV cache 变小),几乎不掉精度。(验收❸口头讲)
  - `rope_theta` → **RoPE(旋转位置编码)**:旋转向量编入位置信息,theta 越大越能外推长文本。(验收❸口头讲)
  - `vocab_size`:词表 ~15 万,对中文友好。

- [ ] **3.2 参数量统计脚本**
  遍历各层 `numel()`,分类汇总(embedding/attention/mlp),算总参数,验证"1.5B"。
  - 要理解:参数主要堆在 MLP;小模型里词表 embedding 占比意外高。
  - 📦 交付:统计脚本 `.py`(+ 输出)。

- [ ] **3.3 Qwen2.5 vs Llama3 架构对比**
  维度:GQA、RoPE、词表、SwiGLU 激活、RMSNorm 归一化、tie embedding 等。
  - 📦 交付:《Qwen2.5 架构分析报告》(`.md` / `.pdf`)。

---

## 阶段 4:Tokenizer 实验(Day 4 全部)🍎

- [ ] **4.1 建 Jupyter Notebook,设计 10+ 极端用例**
  中英混合 / emoji 特殊符号 / 数学公式(∫∑√)/ 超长文本截断 / 连续空格 / 代码片段等。
- [ ] **4.2 观察特殊 token**:`<|endoftext|>`、`<|im_start|>` 的 id 与编解码行为。
- [ ] **4.3 BPE vs SentencePiece 中文分词对比**
  - 要理解:Qwen 用 byte-level BPE,中文常拆成多字节子词;英文 1 词≈1 token,中文 1 字≈1–2 token → "中文更费 token/更贵"。
  - 📦 交付:Tokenizer 实验 Notebook(`.ipynb`,含 10+ 用例)。

---

## ✅ Day 5 已在 Mac 完成(超出原计划)

- [x] LLaMA-Factory 在 Mac MPS 上跑通 **普通 LoRA** 训练(验收❹)—— 无报错
- [x] 两次微调:`alpaca_zh_demo`(3分48秒)、`identity_penguin`(1分39秒)
- [x] 微调前后对比:身份改为 Penguin AI / Ruibo Sun
- [x] TensorBoard 日志产出(`saves/*/runs/`)+ loss 曲线图
- [x] 周报(纯中文)完成

## 🟢 仍需 4090(下周,约半天)

- [ ] `nvidia-smi` 查规格 + 截图
- [ ] 装 CUDA 12.1 版 PyTorch → `torch.cuda.is_available()==True` + 截图(验收❶)
- [ ] 装 vLLM(OpenCompass 可选)
- [ ] 用 **7B QLoRA** 复跑训练(`finetune.py` 改 MODEL=7B、QLORA=4)
- [ ] AWQ 4-bit 压缩(`export_awq.py`)
- [ ] 补齐 nvidia-smi 截图,提交最终报告

---

## 📦 本周交付物总览

| Day | 交付物 | 格式 | 状态 |
|---|---|---|---|
| 1 | 环境截图(conda list + nvidia-smi) | 图片 | conda🍎 / nvidia-smi🟢 |
| 2 | 模型下载截图 | 图片 | 🍎 |
| 2 | 3 组对话日志 | `.md` | 🍎 |
| 2 | 推理脚本 | `inference.py` | 🍎 |
| 3 | 《Qwen2.5 架构分析报告》 | `.md`/`.pdf` | 🍎 |
| 3 | 参数统计脚本 | `.py` | 🍎 |
| 4 | Tokenizer 实验(10+ 用例) | `.ipynb` | 🍎 |
| 5 | 训练跑通日志 + LoRA微调报告 | `.md` + loss图 | ✅🍎 |
| 5 | 《第1周总结报告》 | `.md` | ✅🍎 |

## ✅ 验收标准对照
- ❶ CUDA 可用 → 🟢 4090(Mac 用 MPS)
- ❷ 基座模型能对话 → ✅🍎 已达成
- ❸ 口头解释 RoPE / GQA → ✅🍎 已完成
- ❹ LLaMA-Factory demo 无报错 → ✅🍎 Mac MPS 上 LoRA 训练成功
- ❺ 周报提交 → ✅🍎 已完成(纯中文)
