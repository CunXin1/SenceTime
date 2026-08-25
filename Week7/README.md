# 第 7 周：量化、服务化与应用

> 环境：Windows 11 + RTX 4090 (24GB)。**新增 WSL2 Ubuntu**（vLLM 无 Windows 版）。
> 承接第 4 周的 DPO 交付模型，把「训练好的权重」变成「别人能用的服务」：
> **AWQ/GPTQ 量化 → 三方对比 → vLLM 服务化 → Gradio 前端 → 多模态 → 周报**。
>
> **本周不做任何 finetuning。** 量化是 PTQ（训练后量化），只有前向校准、没有梯度更新；
> 校准集里那 256 篇文档是拿来统计激活分布的，不是训练数据。

## 一、四个关键判断（与任务书的偏差及理由）

### 1. 任务书 34.2 的 `--quantization_method awq` 在 LLaMA-Factory 里做不到

查了本地 LF 0.9.6.dev0 的源码，`quantization_method` 和量化导出是两回事：

| 参数 | 位置 | 实际作用 |
|---|---|---|
| `quantization_method` | `hparams/model_args.py:279` | **加载**已量化的模型（bnb/gptq/awq/hqq…），决定"怎么读别人量化好的权重" |
| `export_quantization_bit` | `model/model_utils/quantization.py:132-164` | **导出**量化模型，这一段写死了 `GPTQConfig` + `GPTQModel` |

`quantization.py:123` 那条 AWQ 分支只做了 `check_version("autoawq")`，属于加载路径。
**LF 的导出只有 GPTQ 一条路，产不出 AWQ 权重。** 因此本周分工：

- **GPTQ** → LLaMA-Factory（`configs/export_gptq_w4.yaml`），满足任务书"用 LF 量化"的要求
- **AWQ** → llm-compressor（`code/quantize_awq.py`）

选 llm-compressor 而不是 AutoAWQ：后者仓库已归档停止维护，官方指向前者；且
llm-compressor 产出的 compressed-tensors 能被 vLLM 原生加载，与 GPTQ 共用一套
oneshot API——Day35 的对比里少掺一层"两个工具的实现差异"。
（Week1 的 `week1/code/finetune/export_awq.py` 那套 AutoAWQ API 留作退路。）

### 2. vLLM 不支持原生 Windows，必须走 WSL2

vLLM 只发 Linux wheel。本机原先没装 WSL，已于 2026-08-13 装上 WSL 2.7.11
（内核 6.18.33.2-2，WSLg + D3D 直通组件就位）。分工：

```
WSL2 Ubuntu               │  Windows
  ~/venvs/quant  量化      │   .venv     Gradio 前端（gradio 5.50.0 已装）
  ~/venvs/vllm   服务      │
  vLLM :8000 (0.0.0.0) ───┼──> WSL2 localhost 转发 ──> Windows 浏览器
```

模型仍放 Windows 侧的 `models/`，WSL 通过 `/mnt/c/...` 访问：不占 WSL 虚拟盘，
路径与前六周一致，代价是首次加载慢几分钟（drvfs I/O），一次性成本。

**WSL 里两个 venv 而不是一个**：vLLM 把 torch 钉死在编译时那个版本（CUDA 扩展与
torch ABI 绑定），而 gptqmodel / llm-compressor / LF 各有独立的 transformers 诉求，
装一起迟早被 pip 解掉。量化和服务本就是先后两个阶段，无需共享运行时——
沿用 Week5 `.venv-vlm` / Week6 `.venv-agent` 的隔离思路。

### 3. 显存降幅必须量**权重**，不能用 nvidia-smi 量进程

vLLM 启动时按 `--gpu-memory-utilization` **预分配** KV cache 把显存吃满。
FP16 与 4-bit 省下的那几 GB，会被 vLLM 拿去多分配等量的 KV cache——
**nvidia-smi 看到的进程显存三种精度完全一样**，直接量必然得出"量化不省显存"的结论。

正确口径是模型权重占用，vLLM 启动日志会打印，`code/bench_quant.py` 负责解析。
量化真正买到的不是"更小的进程"，而是"同样显存里能塞下更多 KV cache / 更长上下文 /
更高并发"——这句话应该写进周报。

### 4. 校准集用自己的领域数据，且与 PPL 评测集严格互斥

AWQ 按**激活幅度**挑显著通道做等价缩放，GPTQ 用校准样本的 Hessian 做逐列误差补偿——
两者的统计量都来自真实前向激活。校准分布偏离部署分布，就会保护错通道。
所以校准集取自 Week4 的偏好数据（`chosen` 侧，即 DPO 后模型实际的输出风格），
而不是 wikitext。

`build_calib_data.py` 先把 1221 条原始样本切成互斥的两池（校准 921 / 留出 300），
再各自拼成长文档：

| 产物 | 条数 | token 长度 (min/中位/max) | 用途 |
|---|---|---|---|
| `data/calib.json` | 256 | 1334 / 1560 / 2570 | AWQ + GPTQ 校准 |
| `data/ppl_eval.json` | 48 | 1333 / 1508 / 2692 | Day35 困惑度评测 |

两个额外约束：

- **必须拼成长文档**。LF 的 `_get_quantization_dataset()`（`quantization.py:60-66`）
  随机抽样后**只接受 token 数严格大于 `export_quantization_maxlen` 的样本**，
  100 次重试全落空就抛 `Cannot find satisfying example`。单条聊天样本只有几百 token，
  直接喂必炸。这里按 1.3× maxlen 拼接并在生成时复核，避免等到导出跑一半才失败。
- **Week3 的 `eval_questions.json` 不进校准集**。那 20 道题是 Day35 测生成质量用的，
  校准看过 = 数据泄漏，量化后 PPL 会虚低。

### 5. 实测环境版本（2026-08-21，WSL2 Ubuntu 24.04）

| venv | 关键包 |
|---|---|
| `~/venvs/vllm` | vllm 0.27.1 / torch 2.13.0+cu130 / compressed-tensors 0.17.0 |
| `~/venvs/quant` | llmcompressor 0.13.0 / GPTQModel 7.3.4 / transformers 5.14.1 / llamafactory 0.9.6.dev0 |

两处安装期修正：

- `gptqmodel` 导入时报 `No module named 'torchvision'`——它的 import 链里带了视觉模型
  分支。补装 `torchvision 0.28.0+cu130`（pip 自动对上 torch 2.13.0+cu130）后正常。
- **待验风险**：LF 0.9.6.dev0 是用 `--no-deps` 装的，pip 没施加版本约束，而当前
  `transformers` 已是 5.14.1（LF 该版本发布时 transformers 还在 4.x）。
  `import llamafactory` 能过，但**导出路径能不能过是另一回事**，Day35 跑 GPTQ 时验证，
  真不兼容就在 quant 环境里单独降 transformers。

## 二、目录结构

```
Week7/
├── code/
│   ├── build_calib_data.py     Day34 校准集 + PPL 留出集（Windows .venv 跑）
│   ├── setup_wsl_vllm.sh       Day34 WSL 双环境搭建 + GPU/架构自检
│   ├── setup_wsl_lf.sh         Day35 第三个 venv：LF 专用（依赖夹击，见第一节）
│   ├── quantize_awq.py         Day34 llm-compressor AWQ W4A16
│   ├── serve_vllm.sh           Day36 起 OpenAI 兼容服务（fp16/awq/gptq/vl 四档）
│   ├── client_demo.py          Day36.2 客户端 Demo（非流式 + 流式 + TTFT）
│   ├── bench_quant.py          Day35 显存/困惑度/吞吐三方对比 → 对比表
│   └── run_bench_all.sh        Day35 三轮「起服务→测→停」的自动编排
├── configs/
│   └── export_gptq_w4.yaml     Day35 LF 的 GPTQ 4-bit 导出
├── data/
│   ├── calib.json              256 篇，量化校准
│   └── ppl_eval.json           48 篇，与校准集不相交
└── deliverables/
    ├── logs/                   serve_*.log（bench 从这里解析权重显存）、setup/export 日志
    ├── bench_{fp16,awq,gptq}.json
    ├── 量化对比表.md            bench_quant.py --report 产出
    └── wsl_env_{vllm,quant}_freeze.txt
```

## 三、执行顺序（2026-08-21 全部实测跑通）

```bash
# ── Windows 侧 ────────────────────────────────────────────────
.venv/Scripts/python.exe Week7/code/build_calib_data.py          # OK

# ── WSL2 Ubuntu 24.04 ─────────────────────────────────────────
bash Week7/code/setup_wsl_vllm.sh        # OK  vllm 0.27.1 / llmcompressor 0.13.0
bash Week7/code/setup_wsl_lf.sh          # OK  LF 专用 venv（第三个）

source ~/venvs/quant/bin/activate
python Week7/code/quantize_awq.py --check                        # OK 抓到静默 bug
python Week7/code/quantize_awq.py                                # OK 4.7 分钟

source ~/venvs/lf/bin/activate
DISABLE_VERSION_CHECK=1 python -m llamafactory.cli export Week7/configs/export_gptq_w4.yaml   # OK 约 8 分钟

bash Week7/code/run_bench_all.sh         # OK 三轮 起服务→测→停 + 汇总
source ~/venvs/vllm/bin/activate
python Week7/code/client_demo.py         # OK 验收②
```

### Day35 实测结果

| 精度 | 权重显存 (GiB) | 相对 FP16 降幅 | 困惑度 (中位) | 相对 FP16 | tok/s (batch=1) | tok/s (并发16) |
|---|---|---|---|---|---|---|
| FP16 | 5.79 | — | 5.231 | — | 119.84 | 1393.81 |
| AWQ  | 1.95 | **66.3%** | 5.639 | +7.80% | **226.54** | **2018.92** |
| GPTQ | 1.94 | **66.5%** | **5.416** | **+3.55%** | 209.50 | 1475.34 |

三条值得写进周报的观察：

1. **显存降幅 66%，远超验收要求的 30%。** 注意 5.79 → 1.94 不是 4 倍——因为
   `lm_head` / embedding（tied，151936×2048 约 0.31B 参数）保持 fp16 未量化，
   只有约 2.6B 的线性层被压到 4-bit。这正是 `quantize_awq.py` 文件头预判的
   「约 2.2GB」，实测 1.95GB。
2. **GPTQ 的质量明显好于 AWQ**（PPL 劣化 +3.55% vs +7.80%），代价是量化耗时长得多
   （约 8 分钟 vs 4.7 分钟）。这与两者的算法差异一致：GPTQ 用 Hessian 做逐列误差补偿
   （**改权重值**去补偿残差），AWQ 只做等价缩放（**改权重尺度**，不补偿残差）。
   多花算力换更小的量化误差，对一次性的离线量化是划算的。
3. **原先「4-bit 在高并发下可能反而更慢」的预判，被数据推翻了。** 并发 16 时
   AWQ 仍有 1.45 倍优势。原因是这个规模下并发 16 远没把 4090 推到算力瓶颈，
   仍在显存带宽区间，反量化开销摊得掉；要复现那个拐点得把并发继续往上推。
   另一个现象：**AWQ 在并发下大幅反超 GPTQ**（2018 vs 1475），而 batch=1 时两者接近
   （226 vs 209）——说明差异来自 kernel 实现（compressed-tensors 走的 Marlin 路径
   对上 GPTQ 的 gptq_v2），不是量化算法本身的差异。

### WSL2 上跑 vLLM 的三个坑（Day36 记录）

服务能起来之前连撞三次，都不是模型问题：

| 症状 | 根因 | 解法 |
|---|---|---|
| `RuntimeError: UVA is not available` | vLLM 0.27.1 默认用 V2 model runner，它依赖 UVA；而 `is_uva_available()` 实为 `is_pin_memory_available()`，vLLM **在 WSL2 上默认关闭 pinned memory**（怕小幅性能回退） | `VLLM_WSL2_ENABLE_PIN_MEMORY=1`（本机内核 6.18.33 >> 门槛 4.19.121） |
| `Could not find nvcc` | 别被误导成"要装 CUDA toolkit"。注意力后端是 FLASH_ATTN（wheel 自带预编译），崩的是**采样器**：FlashInfer 的 top-k/top-p 是 **JIT 编译**的 | `VLLM_USE_FLASHINFER_SAMPLER=0`，退回 PyTorch 原生采样 |
| LF 导出全链报错 | 见第一节：LF 与 gptqmodel 对 transformers 的要求互斥 | 第三个 venv + `DISABLE_VERSION_CHECK=1`，并用实际跑通导出来验证 |

两个 env 已固化进 `serve_vllm.sh`。第二条的性能影响对三种精度**同等施加**，
不影响相对结论，但绝对吞吐值不应拿去和原生 Linux 的数字比——这点要在周报里注明。

## 四、待办与已知风险

- ~~**Day37/38 的 `app.py` 尚未写**~~ —— **2026-08-21 已完成并实测**。
  `code/app.py`：Gradio 流式前端 + 图片上传 + 双后端槽位切换。踩坑两处（都与本机系统代理
  有关，与模型无关）：`openai` SDK 打 127.0.0.1 被注册表代理劫走返 502（解：
  `httpx.Client(trust_env=False)`）；Gradio 启动自检请求自己也被劫走（解：`import gradio`
  之前设 `NO_PROXY`）。证据见 `deliverables/logs/serve_awq.log` 03:16 的 200 记录。
- ~~**Day38 VL 服务未复核**~~ —— **2026-08-25 已复核**。
  `bash Week7/code/serve_vllm.sh vl 8001` 起服务成功（权重 15.67 GiB / 加载 114.6s /
  KV cache 2.3 GiB / 最大并发 5.25×@8K），图文问答字段级正确，TTFT 0.380s。
  日志：`deliverables/logs/serve_vl.log`、`deliverables/logs/vl_smoke.log`。
- ~~**Gemma-4-E4B 大概率不在 vLLM 的 registry 里**~~ —— **2026-08-21 实测推翻**。
  `setup_wsl_vllm.sh` 第 5 步在 vLLM 0.27.1 上查到 367 个已注册架构，其中包含
  `Gemma4ForCausalLM` / `Gemma4ForConditionalGeneration` /
  `Gemma4UnifiedForConditionalGeneration`，以及 `Qwen2_5_VLForConditionalGeneration`。
  **两个 VLM 都能走 vLLM**，不需要再包一层 HF transformers 后端，Day38 的最大工程
  风险消失。保留判断：registry 里有该架构 ≠ E4B 那个 MatFormer/PLE 变体一定能加载成功，
  实际起服务时仍需复核。这条「先查 registry 再排期」的做法本身值得写进周报——
  它把一个原本要占 1-2 天的架构适配风险，压缩成了一条 5 秒钟的自检。
- ~~**Week5 的 VLM 已从本地删除**~~ —— Qwen2.5-VL-7B-Instruct 已重新下载到位（16GB，
  5 分片完整）并跑通服务。gemma-4-E4B-it **未下载**：C 盘只剩 35GB，
  两个 VLM 共约 32GB 放不下，且 Day38 的验收只需要一个多模态后端跑通即可。**不重训 LoRA**。
- **显存编排**：3B-AWQ 常驻（约 2.2GB）+ VLM 占一个互斥槽位（同时只起一个），
  峰值约 18GB / 24GB。三个 fp16 模型共 38GB，装不下。
- ~~**`quantize_awq.py` 需要一次冒烟测试**~~ —— **2026-08-21 已做，且真的抓到了一个静默 bug**。
  llm-compressor 0.13.0 把 AWQ 拆成了两个 modifier：`AWQModifier` 只负责搜索等价缩放
  因子，真正压到 4-bit 的是 `QuantizationModifier`。旧的
  `from llmcompressor.modifiers.awq import AWQModifier` 现在是**废弃 shim，返回的是一个
  list**。原脚本用 `hasattr(recipe, "group_size")` 探测后赋值——对 list 恒为 False，
  于是 `--group-size` **静默失效、不抛任何异常**。已改用新 API 显式构造两段式配方。
  另外查清了 `group_size` 不是 modifier 的字段而在 scheme 里：
  `preset_name_to_scheme("W4A16", ["Linear"])` 本身就是 num_bits=4 / group_size=128 /
  symmetric=True，128 即预设值；偏离 128 才需展开成 `config_groups`。
  **教训**：`hasattr` 式的"兼容性探测"在 API 变动时会把错误变成静默无操作，比直接
  抛异常更难发现——这类地方应该显式断言版本或显式构造。
