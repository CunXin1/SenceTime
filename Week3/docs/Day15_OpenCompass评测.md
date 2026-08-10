# Day15：OpenCompass 通用评测（执行记录）

> 完整操作手册（含 Windows 排障表、兜底命令）：`Week3/code/run_opencompass.md`。
> 交付物：`deliverables/OpenCompass评测分数表.md`。

## 已完成的准备工作（Day12 训练期间并行做完）

| 项 | 状态 |
|---|---|
| 独立环境 `.venv-oc`（不污染训练环境） | ✅ OpenCompass **0.5.3** 安装成功（Python 3.12 + Windows） |
| CEval / CMMLU 数据包 | ✅ OpenCompassData-core 已解压到根目录 `data/`（ceval、cmmlu 在列） |
| `.gitignore` | ✅ 已排除 `/data/`、`.venv-oc/`、`/outputs/` |

## 评测对象（Qwen-only 口径）

1. `models/Qwen2.5-3B-Instruct`（基座）
2. `models/Qwen2.5-3B-week3-best-merged`（最优 SFT，Day16 合并后）

## 执行顺序

```powershell
# ① 冒烟：基座 + ceval_gen，验证能出分（--debug 单进程，Windows 必开）
.\.venv-oc\Scripts\opencompass.exe --datasets ceval_gen --hf-type chat `
  --hf-path models/Qwen2.5-3B-Instruct --max-num-workers 1 --debug

# ② 正式：两个模型 × ceval_gen + cmmlu_gen（顺序跑）
#    结果在 outputs/default/<时间戳>/summary/，手工汇总进分数表
```

若 OpenCompass 在 Windows 上受阻（排障见手册 §4），兜底方案为
`llamafactory-cli eval`（原生 CEval/CMMLU，5-shot MCQA，可直接挂 adapter
无需合并）——分数表中注明所用框架即可。

## 预期与解读口径

- SFT 数据是中文对话指令（4684 条），**预期 CEval/CMMLU 相对基座持平或小幅波动**：
  这类知识型选择题测的是预训练知识，SFT 学的是对话风格。
- 若明显下降（>2 分）：灾难性遗忘信号，需在周报分析（对照 Week2 的
  "代码能力退化"结论，属于同一类现象）。
- OpenCompass 的价值在于给"主观盲测最优"的模型一个**客观通用能力底线**：
  确认调优没有以牺牲通用能力为代价。
