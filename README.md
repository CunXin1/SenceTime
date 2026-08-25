# Qwen2.5-3B 全链路实践：从数据清洗到量化服务

一个 8 周的大模型工程实训项目。以 `Qwen2.5-3B-Instruct` 为基座，在**单张 RTX 4090（24GB）**
上走完 **数据工程 → SFT → DPO 偏好对齐 → 多模态 → Agent → 量化 → 服务化 → 全链路自动化**
的完整闭环，并把每一步的取舍、失败与实测数字都留了档。

**最终交付物**

| 产物 | 说明 |
|---|---|
| `models/Qwen2.5-3B-week4-dpo-merged` | 主力模型：SFT + DPO 对齐后的合并权重（红线拒答率 100%） |
| `models/Qwen2.5-3B-week4-dpo-awq-w4` | AWQ 4-bit 量化版，**权重显存降低 66.3%**，batch=1 吞吐提升 1.89× |
| `models/Qwen2.5-3B-week4-dpo-gptq-w4` | GPTQ 4-bit 量化版，困惑度劣化仅 +3.55% |
| `Week8/reports/` | 8 章技术报告（含全部图表） |
| `run_pipeline.sh` | 一键跑完 数据 → 训练 → 合并 → 评估 → 部署 |

---

## 一、硬件与环境要求

| 项 | 要求 | 本项目实测环境 |
|---|---|---|
| GPU | ≥ 24GB 显存（LoRA 训练 3B + fp16 推理） | RTX 4090 24GB |
| 系统 | Windows 11 / Linux | Windows 11 Pro 26200 |
| Python | 3.12 | 3.12.10 |
| CUDA | 12.4 | 驱动 610.88 |
| 磁盘 | ≥ 60GB（模型权重 + 量化产物 + 数据集） | — |
| **vLLM 服务（可选）** | **必须 Linux 或 WSL2** —— vLLM 不发 Windows wheel | WSL2 Ubuntu 24.04 |

> 显存不够 24GB 时，把 `Week8/configs/sft_best.yaml` 的 `cutoff_len` 降到 1024、
> `per_device_train_batch_size` 降到 1 并同步加大 `gradient_accumulation_steps`（保持等效 batch 不变）。
> `step2_train.sh` 内置的 OOM 重试会自动做这件事，见 §四。

---

## 二、安装

### 2.1 一键创建环境（推荐）

```bash
git clone <repo-url> && cd SenceTime_Weeks1-5

conda env create -f environment.yml
conda activate llm_exp

# LLaMA-Factory 必须单独装，且必须 --no-deps（理由见下）
git clone https://github.com/hiyouga/LLaMA-Factory.git
pip install -e ./LLaMA-Factory --no-deps
pip install omegaconf fire tyro          # --no-deps 漏掉的三个，缺了直接 ImportError
```

> **★ 为什么 LLaMA-Factory 必须 `--no-deps`**
> `requirements.txt` 里的 `datasets 4.0.0` / `accelerate 1.11.0` / `peft 0.18.1` / `trl 0.24.0`
> 四个版本，正好卡在 LF 允许范围的**上界**。直接 `pip install -e ./LLaMA-Factory`
> 会把它们全部降级，而第 1~4 周的实验结论（eval loss 1.2630 等）是在这组版本上跑出来的——
> 降级之后重跑的数字就对不上交付文档了。
> 第 6 周换机重建时正是靠这份锁定把 eval loss 复现到 **1.2630 vs 文档记录的 1.2629**。

### 2.2 用 pip 而不是 conda

```bash
py -3.12 -m venv .venv
.venv/Scripts/activate                    # Windows
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

⚠️ **`--extra-index-url` 不能省**：不加它 pip 会装 CPU 版 torch，训练和评测会慢到不可用
（第 3 周排 OpenCompass 故障时踩过这个坑）。装完先验一句：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.6.0+cu124 True
```

### 2.3 下载模型

```bash
python Week2/code/download_models.py         # Qwen2.5-3B-Instruct，约 5.75GB
```

> **下载慢的时候换源。** ModelScope 实测会掉到 178 kB/s（ETA 2h38m），
> 同一时刻 hf-mirror 有 35 MB/s。`Week5/code/fetch_missing_shard.py` 可以只补下载慢的那个分片。
> 判断标准很简单：**某个分片掉到 1 MB/s 量级，先花 25 秒测另一个源，能省 40 分钟。**

### 2.4 为什么这个项目有五个虚拟环境

不是洁癖，是依赖冲突逼出来的。每一个都对应一次真实的、无法调和的版本互斥：

| 环境 | 位置 | 起因 |
|---|---|---|
| `.venv` | Windows | 主环境：训练 + 评测 + Gradio 前端 |
| `.venv-vlm` | Windows | Gemma-4 需要 `transformers>=5.5`，与主环境的 4.56.2 冲突；升级主环境会打断第 1~4 周的可复现性 |
| `.venv-agent` | Windows | LangChain 会牵动 `transformers`/`tokenizers`/`pydantic` |
| `~/venvs/vllm` | WSL2 | vLLM 把 torch 钉死在编译时那个版本（CUDA 扩展与 torch ABI 绑定） |
| `~/venvs/quant` + `~/venvs/lf` | WSL2 | 量化工具链要新 `transformers`，而 LF 0.9.6 要 4.x —— 两者互斥，只能各占一个环境 |

**只想跑主线（数据 → 训练 → 评估）的话，只需要 `.venv` 一个。**
量化与 vLLM 服务是可选的加分项。

---

## 三、快速开始

```bash
# 冒烟：小步数跑完整链路，验证环境装对了（约 10 分钟，需要 GPU）
bash run_pipeline.sh --quick

# 只做数据准备 + 评估（不训练，不需要长时间占卡）
bash run_pipeline.sh --skip-train --skip-deploy

# 完整跑：数据 → SFT → 合并 → DPO → 合并 → 评估
bash run_pipeline.sh

# 看看现在各段处于什么状态（不执行任何东西）
bash run_pipeline.sh --list
```

> ★ **部署段默认不跑**，需要显式 `--with-deploy`。原因：它起的是常驻服务，
> 会一直占着显存和端口直到被显式停止；一条「跑完就退出」的流水线变成
> 「跑完还挂着两个后台进程」，对无人值守调用是灾难。

分段执行（每一段都能单独跑）：

```bash
# ① 数据：清洗 → SimHash 去重 → 9:1 划分 → 统计报告
.venv/Scripts/python.exe Week8/scripts/step1_data_prep.py

# ② 训练：SFT → 合并 → DPO → 合并（内置 OOM 自动降配重试）
bash Week8/scripts/step2_train.sh --stage all

# ③ 评估：20 题自定义集 5 维自动打分 → CSV（加 --bench 则额外跑 CEval）
.venv/Scripts/python.exe Week8/scripts/step3_eval.py     --model models/Qwen2.5-3B-Instruct --tag base

# ③b 单独跑 CEval（52 学科 1346 题，ppl-5shot 口径，自带评测器）
.venv/Scripts/python.exe Week8/scripts/ceval_local.py     --model models/Qwen2.5-3B-Instruct --tag base

# ④ 部署：WSL 起 vLLM（量化模型）+ Windows 起 Gradio + 健康检查
bash Week8/scripts/step4_deploy.sh --variant awq
bash Week8/scripts/step4_deploy.sh --status
bash Week8/scripts/step4_deploy.sh --stop
```

每个脚本都支持 `--help`；主控还支持 `--dry-run`（只打印将要执行的命令）。

### 3.1 先自检：确认这套东西在你的机器上能跑

```bash
bash Week8/scripts/verify_all.sh          # 快检：不占 GPU，约 1 分钟
bash Week8/scripts/verify_all.sh --full   # 全检：加 GPU 冒烟，约 10 分钟
bash Week8/scripts/verify_all.sh --list   # 只列出会检查哪些项
```

**82 项检查，每一项都真的执行**——文件存在和文件能跑是两回事。覆盖环境版本锁、
脚本语法、YAML 可解析、蒸馏对照组有效性、主控全部参数组合的 dry-run、
三条反向用例（非法参数必须失败）、打分器对齐质量不退化、报告字数与图片完整性、
交付物齐备、仓库无权重文件入库，以及三项 GPU 冒烟。

这是新环境上**第一条该跑的命令**：它失败时给出的是具体哪一项、退出码多少、
输出的最后几行，比让 `run_pipeline.sh` 跑到一半再崩要好定位得多。

> 自检产物写到 `Week8/logs/verify_scratch/`，**不碰正式交付物**。

### 3.2 完整文档

| 文档 | 内容 |
|---|---|
| **[`Week8/README.md`](Week8/README.md)** | Week8 总览：架构、四个关键判断、蒸馏结论、实跑证据 |
| **[`Week8/docs/Pipeline使用说明.md`](Week8/docs/Pipeline使用说明.md)** | 主控参数、四段行为、设计取舍、常见问题 |
| **[`Week8/docs/脚本速查.md`](Week8/docs/脚本速查.md)** | 14 个脚本逐个的用途 / 参数 / 输入输出 / 踩过的坑 |
| [`Week8/docs/Day40_数据与训练自动化.md`](Week8/docs/Day40_数据与训练自动化.md) | 清洗漏斗、去重算法、OOM 五档阶梯、注入式测试 |
| [`Week8/docs/Day41_评估与部署自动化.md`](Week8/docs/Day41_评估与部署自动化.md) | 四级回退、打分器建模、跨 WSL 部署 |
| [`Week8/docs/Day42_知识蒸馏.md`](Week8/docs/Day42_知识蒸馏.md) | 软标签/温度原理、T² 推导、四组对照与显著性 |

---

## 四、目录结构

```
.
├── run_pipeline.sh          一键入口（薄壳，转发到 Week8/scripts/）
├── environment.yml          conda 环境定义
├── requirements.txt         pip 依赖锁定
│
├── models/                  模型权重（gitignored，需自行下载/训练产出）
├── saves/                   LoRA adapter 与训练日志（gitignored）
│
├── week1/  … Week7/         ← 按「周」归档的实验过程（每周自成一体）
│   ├── code/                该周的脚本
│   ├── configs/             该周的训练/导出配置
│   ├── data/                该周的数据
│   ├── docs/                该周的技术笔记
│   └── deliverables/        该周的交付物（含周报）
│
└── Week8/                   ← 按「功能」组织的生产化 Pipeline
    ├── scripts/             step1~4 + 主控 + 蒸馏
    ├── configs/             pipeline.env + 训练/合并/蒸馏配置
    ├── data/                Pipeline 产出的训练/验证集
    ├── logs/                运行日志（gitignored）
    ├── reports/             8 章技术报告 + 图表
    ├── deliverables/        统计报告、评估 CSV、蒸馏对比表
    └── docs/                使用说明与 Day 级工作记录
```

### ★ 关于「按功能模块重组目录」的一处说明

第 8 周的任务书要求把整个项目重组成 `data/ scripts/ configs/ models/ logs/ reports/`。
**这里只在 `Week8/` 内部执行了这个重组，Week1~7 保持按周归档，理由有两条：**

1. **技术原因**：前七周的每个 Python 脚本都用 `Path(__file__).resolve().parents[2]` 定位仓库根。
   把 `Week3/code/x.py` 移到 `weeks/Week3/code/x.py`，`parents[2]` 就会指向 `weeks/` 而不是仓库根，
   **几十个脚本会同时失效**。为了一个目录名去重写全部路径解析，收益与风险不成比例。
2. **文档原因**：七份周报与几十份交付文档里写满了 `Week4/configs/exp/...` 这样的路径引用。
   移动目录会让所有这些引用变成死链，而这些文档正是实验可追溯性的凭证。

**取舍**：历史归档保持稳定，新代码按功能组织。仓库根同时提供 `run_pipeline.sh`、
`environment.yml`、`requirements.txt` 三个功能入口，让「clone 下来第一条命令就能跑」这个
目标不依赖目录结构本身。

---

## 五、各周成果索引

| 周 | 主题 | 关键结论 | 周报 |
|---|---|---|---|
| 1 | 环境与架构 | Mac 上把「下载 → 推理 → 架构分析 → Tokenizer」全流程走通；RoPE / GQA 原理 | [第1周](week1/deliverables/周报/第1周总结报告.md) |
| 2 | 数据工程与 SFT 入门 | 4975 → 4684 条清洗漏斗；**发现代码能力退化**并归因到数据构成 | [第2周](Week2/deliverables/第2周_数据与SFT入门报告.md) |
| 3 | SFT 优化与评估 | 控制变量 4 组实验，最优 `r32/lr1e-4/ep3`（eval loss **1.2630**）；诊断出「Windows 桌面程序抢 GPU 时间片导致训练慢 65%」 | [第3周](Week3/deliverables/第3周_SFT优化与评估报告.md) |
| 4 | DPO 偏好对齐 | β=0.5 红线拒答率 **100%**（β=0.1 仅 85%）；「内容即代码」构造 221 条自建偏好对 | [第4周](Week4/deliverables/第4周_DPO偏好对齐报告.md) |
| 5 | 多模态 | VLM 微调字段准确率 **77.7% → 98.2%**；注意力可视化解释了 OCR 精度差异的机理；**能力更强的模型幻觉更严重** | [第5周](Week5/deliverables/第5周_多模态实践报告.md) |
| 6 | Agent 智能体 | 工具调用 SFT 后端到端 **9/9**；死循环根因定位到「训练数据里 Action Input 与 Observation 不自洽」 | [第6周](Week6/deliverables/第6周_Agent智能体开发报告.md) |
| 7 | 量化与服务化 | AWQ/GPTQ 双路量化，权重显存 **−66%**，KV cache 容量 **+37%**；vLLM + Gradio + 多模态全部跑通 | [第7周](Week7/deliverables/第7周_量化服务化与应用报告.md) |
| 8 | 全链路自动化与蒸馏 | 一键 Pipeline；知识蒸馏探索 | [技术报告](Week8/reports/) |

---

## 六、常见问题

**Q1. `torch.cuda.is_available()` 返回 False。**
八成是装成了 CPU 版 torch。`pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
重装，再验一次。

**Q2. 训练突然慢 5~10 倍，但 GPU 利用率还是 98%，日志里没有任何报错。**
两种可能，都在本项目里真实发生过：
- **桌面程序抢 GPU 时间片**（Windows WDDM 调度）。特征是 GPU 利用率高但**功耗只有额定的 70%**
  （如 333W / 450W）。关掉 Chrome / 微信 / Steam 即可恢复。
- **显存贴顶触发 sysmem fallback**。Windows 上显存不够**不会 OOM 报错**，而是把超出部分放进系统内存，
  静默劣化到 38 倍慢。特征是显存贴着上限、速度掉一个数量级、无任何异常日志。
  解法是留出显存余量（降 `cutoff_len` / `image_max_pixels`），**不是卡着上限跑**。
  ⚠️ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 在 Windows 上是**反效果**，实测更慢，不要跨平台照搬。

**Q3. `llamafactory-cli.exe` 在 Windows 上段错误。**
用 `python -m llamafactory.cli ...` 代替，全项目的配置注释里都写了这一条。

**Q4. 训练报 `CUDA out of memory`。**
`step2_train.sh` 会自动降配重试（batch 减半 + 梯度累积加倍，**等效 batch 保持不变**）。
手动的话按这个顺序降：`per_device_train_batch_size` → `cutoff_len` → `lora_rank`。

**Q5. DataLoader 报奇怪的 CUDA IPC 错误。**
Windows 上 `dataloader_num_workers` 必须为 `0`：spawn 起 worker 会触发 CUDA IPC 误报 OOM。

**Q6. 用 `openai` SDK 打本地 `127.0.0.1:8000` 返回 502，但 `curl` 打同一地址是 200。**
本机系统代理劫持了 localhost 请求。httpx 会读 Windows 注册表里的代理设置，**却不认
`ProxyOverride` 白名单**。解法：`httpx.Client(trust_env=False)`。
如果是 Gradio 自己启动时 502，在 `import gradio` **之前**设 `os.environ["NO_PROXY"]="localhost,127.0.0.1,::1"`。

**Q7. vLLM 在 WSL2 里报 `RuntimeError: UVA is not available`。**
`export VLLM_WSL2_ENABLE_PIN_MEMORY=1`。vLLM 在 WSL2 上默认关闭 pinned memory，而 V2 model runner 依赖它。

**Q8. vLLM 报 `Could not find nvcc`。**
**不是要装 CUDA toolkit。** 崩的是采样器（FlashInfer 的 top-k/top-p 是 JIT 编译的），
注意力后端 FLASH_ATTN 自带预编译 kernel。`export VLLM_USE_FLASHINFER_SAMPLER=0` 即可。

**Q9. 量化之后 `nvidia-smi` 显示显存没变少。**
这是**测量口径问题，不是量化失败**。vLLM 按 `--gpu-memory-utilization` 预分配 KV cache 把显存吃满，
省下的权重显存被拿去多分配 KV cache 了。正确口径是看启动日志的 `Model loading took ... GiB`。
详见技术报告 §6.3——这是本项目最容易得出相反结论的一次测量。

**Q10. 磁盘不够。**
`saves/*/checkpoint-*` 是训练中断续跑用的，训练完成后可以整个删掉
（其中的 adapter 与上级目录的完全一致，实测 byte-identical），本项目清理时释放了 1.76GB。
量化模型（各约 2GB）可以替代 fp16 模型（5.8GB）用于推理。

---

## 七、许可与致谢

基座模型 `Qwen2.5` 系列采用 Apache-2.0 许可。训练框架
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)、推理框架
[vLLM](https://github.com/vllm-project/vllm)、量化工具
[llm-compressor](https://github.com/vllm-project/llm-compressor) 均为开源项目。
本仓库中的代码与文档为实训过程记录。
