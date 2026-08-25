# 第 6 周：Agent 智能体开发

> 环境：Windows 11 + RTX 4090 (24GB)，**独立环境 `.venv-agent/`**（Python 3.12 + LangChain 0.3.30）。
> 承接第 4 周的 DPO 交付模型作为 Agent 的 policy，本周把「会聊天的模型」改造成「会用工具的大脑」：
> **工具实现 → ReAct Agent → 多步推理 → 工具调用 SFT → 错误分析 → 周报**。

## 一、三个关键判断（与任务书的偏差及理由）

### 1. 必须用**文本版** `create_react_agent`

LangChain 里有两个同名函数：`langchain.agents.create_react_agent`（文本版，prompt 驱动）
与 `langgraph.prebuilt.create_react_agent`（要求 native tool calling）。

**Day31.1 要求构造 Thought/Action/Action Input/Observation 格式的训练数据——那正是文本版
ReAct 的格式**，不是 tool-calling 的 JSON 格式。若 Day29 用 tool-calling 版，Day31 微调出来的
格式与 Agent 期望的对不上，整周断成两截。故全周采用文本版。

> ⚠️ **LangChain 1.0（2025-10）已删除 `langchain.agents.create_react_agent`**，v1 只保留
> 基于 tool-calling 的 `create_agent`。因此本周把 langchain 钉在 `>=0.3.0,<1.0.0`。

### 2. 为什么另建 `.venv-agent`

LangChain 会牵动 `transformers`/`tokenizers`/`pydantic`，而主 `.venv` 的 transformers 4.56.2
是 Week1–4 可复现性的地基。沿用 Week5 隔离 `.venv-vlm` 的同一思路。
`.venv-agent` 的 transformers/peft/trl 与主环境同版本，只额外装 LangChain，训练行为一致。

### 3. 知识库是"反向设计"的

先定死 Day30.2 的目标任务（查价格 → 加运费 → 比预算 1000），再造能支撑它的数据：

| 商品 | 价格 | 运费 | 总价 | vs 1000 | 设计意图 |
|---|---|---|---|---|---|
| 星尘 X1 智能音箱 | 899 | 68 | **967** | 未超 | 临界但不超，防止靠"感觉"猜对 |
| 追光 S3 显示器 | 1299 | 120 | **1419** | 超 | 单价已超预算，考察是否仍老实算完 |
| 云雀 Pro 无线耳机 | 459 | **0** | 459 | 未超 | 运费为 0，考察会不会捏造运费 |

## 二、目录结构

```
Week6/
├── code/
│   ├── tools/
│   │   ├── calculator.py       Day28.2 AST 白名单沙箱计算（非 eval）
│   │   ├── knowledge.py        Day28.3 本地 JSON 知识库检索（确定性混合打分）
│   │   └── code_executor.py    Day30.1 AST 语法检查 + 危险节点扫描（不执行）
│   ├── test_tools.py           Day28 三工具单测（31 项，纯 CPU 秒级）
│   ├── local_llm.py            Day29 本地 Qwen 包成 LangChain LLM（prefill + 停止串）
│   ├── agent_react.py          Day29/30 ReAct Agent + 单轮/多步题集
│   ├── build_react_data.py     Day31.1 生成 ReAct 训练数据（每步一条样本）
│   ├── analyze_errors.py       Day32.1 失败模式统计 → 错误分析报告
│   └── rebuild_base_model.ps1  前置：新机器上重建 Week3/Week4 模型链路
├── configs/
│   └── qwen_tool_sft.yaml      Day31.2 工具调用 SFT 配置
├── data/
│   ├── knowledge_base.json     11 条（6 商品 + 5 政策/FAQ）
│   ├── react_sft.json          Day31 训练数据 146 条
│   └── dataset_info.json       LLaMA-Factory 数据集注册（week6_react_sft）
└── deliverables/
    ├── Day28_工具单测日志.md
    ├── agent_run_*.json                 各组 Agent 运行留档（含全部中间步骤）
    ├── Agent错误模式分析报告.md          Day32
    └── 第6周_Agent智能体开发报告.md       周报
```

## 三、运行顺序（在仓库根目录）

```powershell
# ⓪ 前置：新机器需先重建 Day29 的 policy 模型（约 40 分钟 GPU）
#    models/ 与 saves/ 都被 gitignore，换机器后权重全丢，但数据全在 git 里。
powershell -ExecutionPolicy Bypass -File Week6\code\rebuild_base_model.ps1

# ① Day28 三工具单测（纯 CPU，秒级）
.\.venv-agent\Scripts\python.exe Week6\code\test_tools.py --save

# ② Day29 单轮工具调用 / Day30 多步推理
.\.venv-agent\Scripts\python.exe Week6\code\agent_react.py --suite single --model dpo
.\.venv-agent\Scripts\python.exe Week6\code\agent_react.py --suite multi  --model dpo
#    三方对照（Day32 素材）：
.\.venv-agent\Scripts\python.exe Week6\code\agent_react.py --suite all --model base
.\.venv-agent\Scripts\python.exe Week6\code\agent_react.py --suite all --model sft

# ③ Day31 工具调用 SFT（约 5 分钟）
#    ★ 训练用主 .venv（LLaMA-Factory 装在那里），推理才用 .venv-agent。
#      训练是纯 LF 流程不需要 LangChain；两环境 transformers/peft/trl 同版本，adapter 通用。
.\.venv-agent\Scripts\python.exe Week6\code\build_react_data.py
.\.venv\Scripts\python.exe -m llamafactory.cli train Week6/configs/qwen_tool_sft.yaml
.\.venv-agent\Scripts\python.exe Week6\code\agent_react.py --suite all --model dpo `
    --adapter saves\week6\qwen_tool_sft --tag dpo_lora

# ④ Day32 错误模式分析
.\.venv-agent\Scripts\python.exe Week6\code\analyze_errors.py
```

> **Windows 必须用 `python -m llamafactory.cli`**，不要用 `llamafactory-cli.exe`
> ——含撇号的路径（`Ruibo's Desktop`）会让 `.exe` 段错误（Week2 Day10 FAQ）。
>
> **`.ps1` 脚本必须存成 UTF-8 with BOM**：Windows PowerShell 5.1 默认按 ANSI 读 `.ps1`，
> 中文注释会被解码成乱码并撑断字符串终止符。
>
> **训练前关闭 Chrome / 微信 / Steam**：Week3 实测 WDDM 时间片争抢可拖慢 65%。

## 四、三个贯穿全周的技术要点

1. **让 chat 模型「续写自己」**。文本 ReAct 要求模型接着已有的 Thought/Observation 往下写，
   但 Qwen 每个 assistant 轮都从空开始。把 scratchpad 塞进 user 轮会导致模型把它当资料、
   从头再输出一遍 Thought1，**直接死循环**。正确做法是 `apply_chat_template(add_generation_prompt=True)`
   之后把 scratchpad 拼作 **assistant 前缀填充（prefill）**，模型看到"说到一半的自己"自然续写。

2. **训练样本必须在 Action Input 处截断**。若把含 Observation 的完整轨迹当一条样本训，
   模型会学会**自己编造 Observation**（幻觉工具结果）。正确做法是一条 n 步轨迹拆成 n+1 条样本，
   第 k 条以第 k 步的 Action Input 结尾——于是「Action Input 之后就是 EOS」被反复强化。
   这也与推理时的 prefill 精确同分布。

3. **工具描述就是模型选工具的唯一依据**。第三个工具取名 `code_check` 而非 `code_executor`：
   ReAct 里模型极大程度按名字的字面意思选工具，叫 executor 会诱导它拿来求运行结果，
   再把"语法正确"误读成"运行结果"。

## 五、验收标准对照

| # | 验收 | 状态 | 证据 |
|---|---|---|---|
| ❶ | 3 个工具可正常调用 | ✅ | 31/31 单测通过，含 9 项安全攻击全部被拒 |
| ❷ | Agent 能完成至少 3 步的复杂任务 | ✅ | 多步题 **4/4** 严格命中，M1/M2/M3 均为完整 3 步链路 |
| ❸ | 错误分析报告有深度 | ✅ | 5 组对照、7 类失败模式、死循环根因定位到数据 bug 并修复后重测验证 |
| ❹ | 周报提交 | ✅ | `第6周_Agent智能体开发报告.md` |

## 六、最终结果（严格判定口径，9 题 = 5 单轮 + 4 多步）

| 组 | 严格命中 | 多步题 | 失败模式 |
|---|---|---|---|
| `base` 原始基座 | 7/9 | 3/4 | 幻觉 2 |
| `sft` Week3 最优 | 7/9 | 2/4 | 心算 1、幻觉 2 |
| `dpo` Week4 交付 | 8/9 | 3/4 | 选错工具1、心算1、重复调用1、幻觉1 |
| `dpo_lora_v1` 工具SFT（数据有bug） | 7/9 ↓ | 3/4 | **死循环1**、重复调用1、幻觉1 |
| **`dpo_lora` 工具SFT（修复后）** | **9/9** | **4/4** | **无** |

> **本周最有价值的一段**：工具 SFT 第一版反而退步并引入死循环。根因是训练数据里
> `Action Input`（转义的 `\n`）与 `Observation`（用真换行代码算的）不自洽——模型学到
> "发这个输入会得到『语法正确』"，线上却拿到语法错误，于是原样重试成环。
> **而 v1 与 v2 的训练指标几乎完全相同**（loss 0.0132 vs 0.0131），这个 bug 在损失曲线上
> 完全不可见。可复用的原则：**训练数据里的 Observation 必须由 Action Input 里那个
> 一模一样的字符串喂给真实工具产生。**

GPU 总耗时：重建 32m40s + 两次工具 SFT 9m54s ≈ **43 分钟**。
