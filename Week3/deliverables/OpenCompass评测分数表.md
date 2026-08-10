# Week3 Day15 交付：OpenCompass 评测分数表

> 评测框架：OpenCompass 0.5.3（独立环境 `.venv-oc`，Python 3.12 / Windows 11 / RTX 4090）。
> 数据集：CEval（validation）+ CMMLU（test），数据包 OpenCompassData-core-20240207。
> 若某行使用兜底框架（LLaMA-Factory eval，5-shot MCQA），在"评测框架"列注明。

## 总分表

| 模型 | CEval (avg) | CMMLU (avg) | 评测框架 | 备注 |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct（基座） | ⏳ | ⏳ | | 对照基线 |
| Qwen2.5-3B-week3-best-merged（最优 SFT） | ⏳ | ⏳ | | 与基座差值：⏳ |

## 分学科明细（可选，从 summary csv 摘录大类）

| 模型 | STEM | 社科 | 人文 | 其他 |
|---|---|---|---|---|
| 基座 | ⏳ | ⏳ | ⏳ | ⏳ |
| 最优 SFT | ⏳ | ⏳ | ⏳ | ⏳ |

## 结论

⏳（评测完成后填写：SFT 是否保住了通用能力底线；若有明显涨跌，
结合训练数据构成解释原因，对照 Week2"代码能力退化"分析框架。）

## 原始输出位置

- OpenCompass：`outputs/default/<时间戳>/summary/`（gitignored，分数以本表为准）
- 运行命令与排障记录：`Week3/code/run_opencompass.md`
