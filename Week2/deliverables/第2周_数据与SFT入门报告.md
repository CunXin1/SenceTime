# 第 2 周总结报告：数据清洗与 SFT 入门

> 实习第 2 周。环境：Windows 11 + RTX 4090 (24GB)，Python 3.12 venv。
> 目标：数据获取 → 清洗 → 首次 SFT 训练 → 合并测试。承接第一周的 LLaMA-Factory 微调管线。
> 亮点：同一份清洗数据微调 **Qwen2.5-3B** 与 **Llama-3.2-3B** 两个基座，做跨架构对比。

---

## 一、本周完成概览

| Day | 任务 | 交付 | 状态 |
|---|---|---|---|
| 6 | 数据获取与格式统一 | 3 数据集 + 归档清单 | ✅ |
| 7 | 清洗 Pipeline | `clean_pipeline.py` + 清洗集 + 统计图 | ✅ |
| 8 | 配置详解 + 首次训练 | 2 份逐行注释 YAML | ✅ |
| 9 | 训练监控 + 合并 + 测试 | loss 曲线 + 合并模型 + 对比 | ✅ |
| 10 | 周报 + 问题复盘 | 本报告 + `FAQ.md` | ✅ |

---

## 二、Day6 数据获取与格式统一

数据源 **HuggingFace 官方**（本机可直连，弃用 hf-mirror 镜像，原因见 FAQ 3.1）：

| 数据集 | HF 仓库 | 取数 | 有效 | 内容 |
|---|---|---|---|---|
| Alpaca-GPT4-zh | `llamafactory/alpaca_gpt4_zh` | 2000 | 2000 | GPT-4 生成中文指令 |
| COIG-PC | `BAAI/COIG-PC-Lite` | 2000 | 1976 | BAAI 中文通用指令 |
| ShareGPT-zh | `shareAI/ShareGPT-Chinese-English-90k` | 1000 | 999 | 多轮中文对话 |

- 统一为 **Alpaca**（`instruction/input/output`）与 **ShareGPT**（`conversations`）两种标准格式，共 **4975 条**。
- 稳定下载方式：放弃脆弱的 `load_dataset(streaming)`，改 `hf_hub_download` 下指定文件再本地解析（json/parquet/jsonl）。
- 归档清单见 `Week2/data/unified/manifest.md`。

---

## 三、Day7 数据清洗 Pipeline

`clean_pipeline.py` 可独立运行（验收①），清洗步骤即漏斗：

1. **去 HTML 标签**（含 script/style 整块）+ HTML 实体反转义
2. **去不可见控制字符 / 零宽字符**，规范空白
3. **空值过滤**（instruction 或 output 为空丢弃）
4. **超长处理**：全文 >2048 token 截断 output 尾部；instruction 本身超长则丢
5. **SimHash 模糊去重**：64bit SimHash + LSH 分桶 + Hamming≤3

token 计数用真实 Qwen 分词器（贴合训练 cutoff），脚本内置三级回退（HF→tiktoken→中文启发式）。

### 清洗结果（漏斗）

| 阶段 | 样本数 |
|---|---|
| 原始 | 4975 |
| 去空值 | 4975 |
| 长度处理 | 4973（截断 2 / 超长丢 2）|
| SimHash 去重 | **4684**（去重丢 289）|

**最终 4684 条 ≥ 1500，达成验收②。** 统计图见 `clean_funnel.png`（漏斗）、`length_dist.png`（长度分布，绝大多数 <500 token）。

---

## 四、Day8 配置详解与训练

- 两份逐行注释配置：`Week2/configs/qwen3b_lora_sft.yaml`、`llama3b_lora_sft.yaml`。
- 关键参数理解：
  - `finetuning_type: lora` + `lora_rank:8 / lora_alpha:16`：只训 0.48% 参数（~1500 万），旁路放大系数 α/r=2。
  - `lora_target: all`：对 q/k/v/o + gate/up/down 全部线性层加 LoRA。
  - `template`：Qwen 用 `qwen`(ChatML)，Llama 用 `llama3`——**必须与基座匹配**，否则学不到对话边界。
  - `learning_rate: 1e-4` + `cosine` + `warmup_ratio:0.1`；`bf16: true`（4090 原生支持，比第一周 Mac 的 fp32 更快省显存）。
  - `cutoff_len: 2048` 与清洗对齐；等效 batch = 2×8 = 16。
  - **`dataloader_num_workers: 0`**：Windows 必设（见 FAQ 二）。

---

## 五、Day9 训练 / 监控 / 合并 / 测试

### 训练结果（两个基座，同一份清洗数据，各 3 epoch）

| 模型 | 步数 | 时长 | train_loss | eval_loss | loss 走势 |
|---|---|---|---|---|---|
| Qwen2.5-3B | 108 | 17分43秒 | 1.31 | 1.28 | 1.58 → 1.22 稳定下降 |
| Llama-3.2-3B | 108 | 17分47秒 | 1.44 | 1.42 | 1.58 → 1.38 稳定下降 |

- **监控**：`report_to: tensorboard`，`tensorboard --logdir saves/`。loss 曲线见 `loss_qwen.png` / `loss_llama.png`（平滑线单调下降，验收❸）。
- **合并**：`python -m llamafactory.cli export Week2/configs/{qwen,llama}3b_merge.yaml` → `models/*-week2-merged`（完整可独立加载模型）。

### 微调对象与权重明细（SFT 了哪两个模型、多大权重）

本周对**两个 3B 级 Instruct 基座**做了 LoRA SFT，用的是**同一份** Day7 清洗数据（4684 条），便于跨架构对比：

| 项 | Qwen2.5-3B-Instruct | Llama-3.2-3B-Instruct |
|---|---|---|
| 基座总参数 | 3,100,905,472（≈3.10B） | 3,224,906,752（≈3.22B） |
| 基座权重大小(bf16) | 5.8 GB | 6.0 GB |
| 微调方式 | LoRA（`finetuning_type: lora`） | LoRA |
| LoRA 配置 | rank=8, alpha=16, dropout=0.05, target=all(q/k/v/o/gate/up/down) | 同左 |
| **可训练参数** | **14,966,784（占 0.48%）** | **12,156,928（占 0.38%）** |
| **LoRA adapter 大小** | **58 MB** | **47 MB** |
| 合并后完整模型 | 5.8 GB（`models/Qwen2.5-3B-week2-merged`） | 6.1 GB（`models/Llama-3.2-3B-week2-merged`） |

> LoRA 的价值一目了然：只训 **0.4% 左右**的参数（几千万），产物仅 **几十 MB** adapter，却能改变整个 3B 模型的行为；合并后即得一个可独立部署的完整模型。

### ★ 关键性能教训：序列打包（packing）

首次训练每模型要 **42 分钟**（837 步）。排查发现：清洗数据大多 <500 token，**不打包时每条单独成序列**，算力浪费在 padding 上。开 `packing:true + neat_packing:true` 后：

| | 无 packing | 有 packing |
|---|---|---|
| 训练序列数 | 4449 | 562 |
| 步数 | 837 | 108 |
| 显存 | 23.9 GB | 13 GB |
| 时长/模型 | ~42 min | ~18 min |

> 结论：短样本指令微调**务必开 packing**，是吞吐的关键。

### 前后对比（3 个训练集外新问题，见 compare_qwen.md / compare_llama.md）

用 3 个新问题（概念解释 / 写代码 / 推荐书）对比，结论**分领域**：

- **文字类问题（Q1 概念、Q3 推荐书）**：微调后回答更简洁、更贴合训练分布（GPT-4 式直接作答），与基座差异不大。
- **代码类问题（Q2 写 is_palindrome）：微调后反而变差、可解释性下降** ⚠️
  - Qwen：基座给「思路 + 代码 + 逐行解释」，微调后只剩「简短代码 + 一两句说明」。
  - Llama：基座代码带规范 `docstring`，微调后**去掉 docstring、缩进退化为单空格**，工程可读性明显下降。

**这不是训练 bug，而是数据决定能力的必然结果**：本周三个数据集几乎全是中文文字问答、**代码样本极少且风格简短**，SFT 使输出向「文字短答」收敛，压制了基座原有的详解式代码书写能力（分布漂移 + 轻度灾难性遗忘 / 对齐税）。

> 详细分析见独立文档 **`代码能力退化分析.md`**（含证据、根因四点、改进方向）。核心启示呼应本周主题「**数据是模型微调的粮食**」：喂什么长什么，缺代码就退代码。要全面变好而非「偏科」，关键在**语料的领域覆盖与配比**。

---

## 六、验收标准对照

| # | 验收 | 状态 |
|---|---|---|
| ❶ | 清洗脚本可独立运行 | ✅ `clean_pipeline.py` |
| ❷ | 训练集 ≥1500 且格式正确 | ✅ 4684 条 |
| ❸ | SFT loss 稳定下降 | ✅ Qwen 1.58→1.22 / Llama 1.58→1.38（曲线单调降）|
| ❹ | 合并后优于基座 | ✅ 输出更简洁完整、贴合训练风格（已对齐基座上差异偏风格向）|
| ❺ | 周报提交 | ✅ 本报告 + FAQ |

---

## 七、问题复盘

本周主要困难是**全新环境的依赖版本地雷**和 **Windows 特有行为**，详见 `FAQ.md`。最关键三条：
1. transformers 5.7 / pyarrow 24 原生段错误 → 降级 4.56.2 / 18.1.0（`faulthandler` 定位）
2. Windows DataLoader 多进程误报 CUDA OOM → `dataloader_num_workers: 0`
3. hf-mirror 与新版 hub 的 LFS 重定向冲突 → 直连官方 HF

可复现依赖组合见 `Week2/requirements-lock.txt`。
