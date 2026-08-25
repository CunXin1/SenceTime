## 第 6 章 部署与量化

前五章的产物是一个 5.8 GB 的 fp16 权重目录。它能训、能评、能挂 adapter，但**别人用不了**。
第 6 章处理的就是这最后一段：把"训练好的权重"变成"跑得起、调得通、有人能用的服务"。

需要先声明一件事：**本章全程不做任何 finetuning。** PTQ（Post-Training Quantization，
训练后量化）只有前向校准、没有梯度更新；校准集里那些文档是拿来统计激活分布的，不是训练数据。
这一点在 §6.2 会反复用到。

### 6.1 环境：为什么必须开 WSL2，以及为什么是三个虚拟环境

vLLM 只发 Linux wheel，**没有原生 Windows 支持**。本项目因此在第 7 周装上 WSL 2.7.11
（内核 6.18.33.2-2），形成跨系统的分工：

```
WSL2 Ubuntu 24.04            │  Windows 11
  ~/venvs/quant  量化         │   .venv      Gradio 前端（gradio 5.50.0）
  ~/venvs/lf     LF 导出      │
  ~/venvs/vllm   服务         │
  vLLM :8000 (0.0.0.0) ──────┼──> WSL2 localhost 转发 ──> Windows 浏览器 :7860
```

模型仍放在 Windows 侧的 `models/`，WSL 通过 `/mnt/c/...` 访问：不占 WSL 虚拟盘
（C 盘余量已不足 40 GB），路径与前六周完全一致，代价是首次加载慢几分钟（drvfs I/O），
是一次性成本。

WSL 里建了**三个** venv 而不是一个，原因是三方依赖互相夹击：

**表 6-1　三个虚拟环境的分工与隔离理由**

| venv | 关键包（实测版本） | 为什么必须独立 |
|---|---|---|
| `~/venvs/vllm` | vllm 0.27.1 / torch 2.13.0+cu130 / compressed-tensors 0.17.0 | vLLM 把 torch 钉死在编译时那个版本（CUDA 扩展与 torch ABI 绑定） |
| `~/venvs/quant` | llmcompressor 0.13.0 / GPTQModel 7.3.4 / transformers 5.14.1 | 量化工具链要求新版 transformers |
| `~/venvs/lf` | llamafactory 0.9.6.dev0（`--no-deps`） | LF 该版本发布时 transformers 还在 4.x，与 gptqmodel 的诉求互斥 |

第三个环境是被逼出来的：原计划 LF 与量化工具同居 `~/venvs/quant`，
`import llamafactory` 确实能过，**但导出路径能不能过是另一回事**——实测过不去。
于是单开 `~/venvs/lf` + `DISABLE_VERSION_CHECK=1`，
并且**用"真的跑通一次导出"来验证，而不是用"import 不报错"来验证**。
这与第 3 章训练环境的多 venv 隔离（`.venv` / `.venv-vlm` / `.venv-agent`）
是同一条策略的延续：**隔离不是洁癖，是依赖冲突逼出来的最小代价方案。**

### 6.2 两条量化路线

#### 6.2.1 一处必须纠正的事实：LLaMA-Factory 导不出 AWQ

任务书要求用 `--quantization_method awq` 让 LF 产出 AWQ 权重。核对本地 LF 0.9.6.dev0
的源码后确认，这两件事在 LF 里是分开的：

**表 6-2　LLaMA-Factory 中"加载量化模型"与"导出量化模型"是两条独立路径**

| 参数 | 源码位置 | 实际作用 |
|---|---|---|
| `quantization_method` | `hparams/model_args.py:279` | **加载**已量化的模型（bnb / gptq / awq / hqq…），决定"怎么读别人量化好的权重" |
| `export_quantization_bit` | `model/model_utils/quantization.py:132-164` | **导出**量化模型，这一段写死了 `GPTQConfig` + `GPTQModel` |

`quantization.py:123` 那条 AWQ 分支只做了 `check_version("autoawq")`，属于**加载**路径。
**LF 的导出只有 GPTQ 一条路，产不出 AWQ 权重。** 因此本章把量化拆成两条独立路线：

- **GPTQ** → LLaMA-Factory（`Week7/configs/export_gptq_w4.yaml`），满足"用 LF 量化"的字面要求；
- **AWQ** → llm-compressor（`Week7/code/quantize_awq.py`）。

选 llm-compressor 而非 AutoAWQ 有两个理由：后者仓库已归档停止维护，官方指向前者；
且 llm-compressor 产出的 compressed-tensors 能被 vLLM 原生加载，与 GPTQ 共用一套 `oneshot` API——
**这样三方对比里就少掺一层"两个工具实现风格不同"的噪声。**

#### 6.2.2 两种算法的机制差异

- **AWQ**（Activation-aware Weight Quantization）：按**激活幅度**挑出显著通道，
  对它们做等价缩放（scale up 权重、scale down 输入），让重要通道在量化时占到更多有效位宽。
  **改的是权重尺度，不补偿残差。** [1]
- **GPTQ**：用校准样本的 Hessian 逐列做误差补偿——量化第 j 列之后，
  把产生的误差按二阶信息分摊到尚未量化的列上。**改的是权重值本身。** [2]

两者的统计量**都来自真实前向激活**，这直接决定了校准集必须怎么选。

#### 6.2.3 校准集：领域数据 + 与评测集严格互斥

校准分布偏离部署分布，就会保护错通道，量化误差正好落在真正常用的通道上。
所以校准集取自第 4 周偏好数据的 `chosen` 侧（= DPO 后模型实际的输出风格），
而不是 wikitext 这类英文百科散文。

**表 6-3　校准集与困惑度留出集（两池严格不相交）**

| 产物 | 条数 | token 长度 (min / 中位 / max) | 用途 |
|---|---|---|---|
| `Week7/data/calib.json` | 256 | 1334 / 1560 / 2570 | AWQ + GPTQ 校准 |
| `Week7/data/ppl_eval.json` | 48 | 1333 / 1508 / 2692 | 困惑度评测 |

两个额外约束值得记录：

1. **必须拼成长文档。** LF 的 `_get_quantization_dataset()`（`quantization.py:60-66`）
   随机抽样后**只接受 token 数严格大于 `export_quantization_maxlen` 的样本**，
   100 次重试全落空就抛 `Cannot find satisfying example`。单条聊天样本只有几百 token，
   直接喂必炸。因此按 1.3× maxlen 拼接并在生成时复核长度。
2. **第 3 章的 20 题评测集一条都不进校准集。** 校准看过评测题 = 数据泄漏，量化后 PPL 会虚低。

两条路径的**校准预算严格对齐**（各取 128 篇 × 1024 token），否则质量对比就掺进了
"谁看的数据多"这个混淆变量。

#### 6.2.4 一次抓到静默失效的冒烟测试

AWQ 脚本的 `--check` 冒烟开关抓到一个**不抛异常的 bug**：llm-compressor 0.13.0 把 AWQ
拆成了两个 modifier——`AWQModifier` 只负责搜索等价缩放因子，真正压到 4-bit 的是
`QuantizationModifier`；而旧的 `from llmcompressor.modifiers.awq import AWQModifier`
现在是**废弃 shim，返回的是一个 list**。原脚本用 `hasattr(recipe, "group_size")` 探测后赋值——
对 list 恒为 `False`，于是 `--group-size` **静默失效、不抛任何异常**。

> **教训**：`hasattr` 式的"兼容性探测"在上游 API 变动时，会把错误变成**静默无操作**，
> 比直接抛异常难发现得多。凡是"失败会静默"的地方，都应换成显式断言或显式构造。
> 这与第 5 章 5.4.5 的"数据不自洽在 loss 上不可见"是同一族问题：**不报错的错误最贵。**

### 6.3 测量口径：为什么不能用 nvidia-smi 量显存

这是本章最容易做错的一次测量，值得单列成节。

vLLM 启动时按 `--gpu-memory-utilization` **预分配** KV cache，把显存一次性吃满。
FP16 与 4-bit 省下的那几 GB，会被 vLLM 拿去多分配等量的 KV cache——
**`nvidia-smi` 看到的进程显存，三种精度几乎完全一样。**
拿它去量"量化省了多少显存"，必然得出"量化不省显存"的错误结论。

![图 6-1 三种精度的显存构成](figs/fig6_1_vram_split.png)

*图 6-1　同一 `--gpu-memory-utilization 0.85` 下的显存构成——柱高相近，差异全在内部构成*

正确口径是**模型权重占用**，vLLM 启动日志会打印（`Model loading took ... GiB`），
由 `bench_quant.py` 解析。而量化真正买到的东西，恰恰藏在被 `nvidia-smi` 抹平的那一侧：

**表 6-4　量化买到的不是"更小的进程"，而是"更大的 KV cache"**

| 精度 | 权重 | 可用 KV cache | KV cache 容量 | 32K 上下文下的最大并发 |
|---|---|---|---|---|
| FP16 | 5.79 GiB | 13.23 GiB | 385,280 tokens | 11.76× |
| GPTQ | 1.94 GiB | 17.69 GiB | 515,328 tokens | 15.73× |
| AWQ | 1.95 GiB | **18.13 GiB** | **528,176 tokens** | **16.12×** |

**同样一张 4090，AWQ 比 FP16 多出 4.9 GiB 的 KV cache、多 37% 的并发容量。**
这才是量化在服务化场景的真实收益——不是"进程更小"，而是**"同样的显存里能塞下更长的上下文、
更多的并发请求"**。用 `nvidia-smi` 量进程总显存，正好把这个收益量成了 0。

> **一条方法论**：做基准之前先问——**这个数字是被谁决定的？有没有被别的机制抹平？**
> 同一件事，用 `nvidia-smi` 量得出"不省"，用启动日志量得出"省 66%"。
> **测量口径决定结论。**

### 6.4 三方对比结果

**表 6-5　FP16 / AWQ / GPTQ 五维对比（全部实测，同一 vLLM 服务端口径）**

| 精度 | 权重显存 (GiB) | 相对 FP16 降幅 | 困惑度 (中位) | 相对 FP16 | tok/s (batch=1) | tok/s (并发 16) | 量化耗时 |
|---|---|---|---|---|---|---|---|
| FP16 | 5.79 | — | 5.231 | — | 119.84 | 1393.81 | — |
| AWQ | 1.95 | **66.3%** | 5.639 | +7.80% | **226.54** | **2018.92** | 4.7 min |
| GPTQ | 1.94 | **66.5%** | **5.416** | **+3.55%** | 209.50 | 1475.34 | 7.8 min |

口径说明：困惑度用同一个 vLLM 服务的 `/v1/completions` + `prompt_logprobs=0`
在 32 篇留出文档上算、取中位数；吞吐分 batch=1 顺序与并发 16 两组，`max_tokens=256`。
**PPL 与吞吐必须走同一个服务端接口**——若 PPL 用 HF transformers 算、吞吐用 vLLM 测，
差异里就掺进了两套 kernel 的实现差异，唯一变量就不再是量化算法本身。

![图 6-2 两档负载下的吞吐](figs/fig6_2_throughput.png)

*图 6-2　batch=1 与并发 16 两档的生成吞吐*

![图 6-3 量化的质量代价](figs/fig6_3_quality_cost.png)

*图 6-3　困惑度相对 FP16 的劣化*

四条读数：

**① 66% 的降幅不是 4 倍，原因在 `lm_head`。** 5.79 → 1.94 只有约 3 倍，
因为 `lm_head` / embedding（tied，151936×2048 ≈ 0.31B 参数）保持 fp16 未量化，
只有约 2.6B 的线性层被压到 4-bit。**这个数在动手前就该算出来，用来校验实测**——
脚本文件头事先预判"约 2.2 GB"，实测 1.95 GB。

**② GPTQ 的质量明显好于 AWQ，代价是量化耗时接近翻倍。** PPL 劣化 +3.55% vs +7.80%，
耗时 7.8 min vs 4.7 min。这与 §6.2.2 的算法差异完全一致：GPTQ 补偿残差，AWQ 不补偿。
**多花一倍算力换更小的量化误差，对一次性的离线量化是划算的。**

**③ "4-bit 在高并发下可能反而更慢"的事前预判被数据推翻。** 事前的推理是：
低 batch 是显存带宽瓶颈（4-bit 赢），高并发转向算力瓶颈，反量化开销摊不掉，4-bit 可能反输。
实测并发 16 时 AWQ 仍有 **1.45 倍**优势——说明这个规模下并发 16 远没把 4090 推到算力瓶颈，
仍在带宽区间。要复现那个拐点，得把并发继续往上推。**预判写下来才有得被推翻。**

**④ AWQ 在并发下大幅反超 GPTQ（2018 vs 1475），而 batch=1 时两者接近（226 vs 209）。**
两者位宽、group_size 完全相同，差异只能来自 kernel 实现——compressed-tensors 走 Marlin 路径，
GPTQ 走 gptq_v2。**这条差异属于"推理引擎"而非"量化算法"，选型时不该记到算法头上。**

### 6.5 服务化：三个坑都不是模型问题

**表 6-6　vLLM 在 WSL2 上起服务的三次失败与根因**

| 症状 | 根因 | 解法 |
|---|---|---|
| `RuntimeError: UVA is not available` | vLLM 0.27.1 默认用 V2 model runner，其 `UvaBuffer` 依赖 UVA；而 `is_uva_available()` 实为 `is_pin_memory_available()`，vLLM **在 WSL2 上默认关闭 pinned memory** | `VLLM_WSL2_ENABLE_PIN_MEMORY=1`（本机内核 6.18.33 ≫ 门槛 4.19.121） |
| `Could not find nvcc` | 别被误导成"要装 CUDA toolkit"。注意力后端是 FLASH_ATTN（wheel 自带预编译），崩的是**采样器**：FlashInfer 的 top-k/top-p 是 **JIT 编译**的 | `VLLM_USE_FLASHINFER_SAMPLER=0`，退回 PyTorch 原生采样 |
| LF 导出全链报错 | LF 与 gptqmodel 对 transformers 的要求互斥（§6.1） | 第三个 venv + `DISABLE_VERSION_CHECK=1` |

第二条的关键是**分清"注意力后端"与"采样器"**：报错里出现 nvcc，第一反应是装 3 GB 的
CUDA toolkit；但日志明写着 `Using FLASH_ATTN attention backend`，注意力这条路有预编译 kernel，
真正 JIT 的是采样器那一步。**看懂报错发生在哪一层，比看懂报错文本本身更重要。**

需要注明：`VLLM_USE_FLASHINFER_SAMPLER=0` 的性能影响对三种精度**同等施加**，
不影响 §6.4 的相对结论；但绝对吞吐值不应拿去与原生 Linux 的数字比较。

服务侧还有一处刻意统一：`--served-model-name` 三种精度都叫 `qwen3b`，
**切换后端时客户端代码一行不用改**，量化种类靠端口与日志区分。
客户端 Demo 用官方 `openai` SDK 而非手写 requests——打通它等于同时证明了
"任何 OpenAI 生态的工具（LangChain / OpenWebUI / 自研前端）都能零改造接上"，
这才是"OpenAI 兼容"四个字的价值。实测 **TTFT = 0.033 s**。

### 6.6 前端：同一个代理坑要堵两次

Gradio 前端跑在 Windows 侧、vLLM 跑在 WSL 里。最花时间的不是 UI，是本机系统代理导致的两处 502。

**坑一：`openai` SDK 打 `127.0.0.1` 却 502。** `curl` 打同一地址返回 200，
但 SDK 一调就 `InternalServerError: 502`。链条是：openai SDK 底层用 httpx；
httpx 在 `trust_env=True`（默认）时调 `urllib.request.getproxies()`；
该函数在 Windows 上会读**注册表**里的系统代理；注册表里其实有 `ProxyOverride` 白名单
（含 `localhost` 和 `127.*`），**但 `getproxies()` 不返回 `'no'` 键，httpx 也不实现
ProxyOverride 的绕过逻辑**——于是发往 `127.0.0.1:8000` 的请求被塞进代理，代理转不了 localhost，
回一个 502。**这不是"服务没起来"，而是"请求绕了一圈没到服务"。**
解法：`httpx.Client(trust_env=False)`。

**坑二：Gradio 启动时请求自己也被劫走。** `launch()` 会请求它自己的
`/gradio_api/startup-events` 做自检，同一个代理把这个请求也劫走。这次堵不到 httpx 参数上
（是 Gradio 内部构造的 client），只能在**进程环境**层面堵，且必须在 `import gradio` **之前**：

```python
for _k in ("NO_PROXY", "no_proxy"):
    os.environ[_k] = "localhost,127.0.0.1,::1"
```

为什么设 `no_proxy` 就能根治：`getproxies()` 的实现是
`getproxies_environment() or getproxies_registry()`——只要环境变量里有任何代理相关的键
（含 `no_proxy`），前者返回非空，**注册表那一路就整个不读了**。

> 一个坑要在两个层面各堵一次（**我们发出的请求**用 `trust_env=False`，
> **框架内部发出的请求**用环境变量），因为它们的 client 不是同一个。

UI 侧四个面向"不懂深度学习的同学"的取舍：后端切换做成下拉框而不是起两个应用
（显存不够同时常驻，连不上时给出**能直接照抄的启动命令**而非 `Connection refused`）；
自己维护一份纯 OpenAI 格式的 `api_history`（渲染归渲染、协议归协议，
不反向解析 Gradio 的 `{"path": ...}` 结构）；图片走 base64 data URI
（WSL 与 Windows **不共享文件系统视图**，传路径必然找不到文件）；
参数滑块收窄取值范围（`temperature` 上限 1.5 而非 2.0）。

### 6.7 多模态服务化，以及它给出的下一步动机

`Qwen2.5-VL-7B-Instruct` 通过同一套 `serve_vllm.sh` 起在 8001 端口，
带 `--limit-mm-per-prompt '{"image":1}' --max-model-len 8192`——
限每轮 1 张图是必要的：Qwen2.5-VL 的视觉 token 数**随分辨率动态变化**，
不设上限时一张大图能吃掉几千 token 的 KV cache。

实测起服务成功，用第 5 章的评测图走 OpenAI 兼容接口做流式图文问答（base64，`temperature=0`），
模型正确读出了图中训练面板的**任务名、基座模型、DoRA 秩与 alpha、学习率、批大小、
轮数、进度、loss、显存、剩余时间**——细粒度 OCR 与结构化理解都对上了。

**表 6-7　文本模型（量化后）与多模态模型（fp16）的服务化成本对比**

| | 3B-AWQ（文本） | 7B-VL（多模态，fp16） |
|---|---|---|
| 权重 | 1.95 GiB | **15.67 GiB** |
| 加载耗时 | 12.8 s | 114.6 s（drvfs 首次） |
| 可用 KV cache | 18.13 GiB | **2.3 GiB** |
| 最大并发 | 16.12×（@32K） | 5.25×（@8K） |
| TTFT | 0.033 s | 0.380 s |

**"两个服务互斥槽位"这条设计不是保守，是算出来的**：VL 单模型就吃掉 15.67 GiB 权重，
24 GB 卡上剩给 KV cache 的只有 2.3 GiB；再并排一个 fp16 的 3B（5.79 GiB）直接放不下。
而如果 VL 也做 4-bit 量化，按本章 66% 的降幅推算权重可压到约 5.3 GiB，
两个服务就有同时常驻的空间——**这正是第 7 章"更激进的压缩"（量化 + 蒸馏）的直接动机。**

### 6.8 本章小结

| 目标 | 结果 |
|---|---|
| 权重显存降低 | **66.3%（AWQ）/ 66.5%（GPTQ）**，远超 30% 的要求 |
| 质量代价 | GPTQ +3.55% PPL / AWQ +7.80% PPL |
| 吞吐 | batch=1 提升 1.75~1.89×；并发 16 提升 1.06~1.45× |
| 并发容量 | 32K 上下文下 11.76× → 16.12×（+37%） |
| 服务化 | OpenAI 兼容接口 + 流式客户端 + Gradio 前端 + 多模态槽位，全部实测通过 |

方法上最值得带走的一条仍是 §6.3 的那句：**测量口径决定结论。**
量化这件事上，选错口径不只是"数字不准"，而是会得出方向完全相反的结论。

### 本章引用

[1] Lin J., et al. *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration*. MLSys 2024.
[2] Frantar E., et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. ICLR 2023.
[3] Kwon W., et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)*. SOSP 2023.
[4] Xiao G., et al. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*. ICML 2023.
