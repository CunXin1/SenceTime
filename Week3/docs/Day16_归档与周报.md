# Day16：最优模型归档与周报（操作清单）

> 交付物：`models/Qwen2.5-3B-week3-best-merged/`、《最优模型说明.md》、
> 《第3周_SFT优化与评估报告.md》。

## 前置条件

Day12-15 的三重证据齐备：实验结果汇总（eval loss）、盲测得分汇总（加权总分）、
OpenCompass 分数表（通用能力底线）。按《实验计划表.md》§6 判定规则选出最优组。

## 操作步骤

```powershell
# ① 把 Week3/configs/merge_best_qwen.yaml 的 adapter_name_or_path
#    改成最优组的 saves/week3/qwen/<best_run>

# ② 合并导出（Windows 用 python -m，llamafactory-cli.exe 会段错误）
.\.venv\Scripts\python.exe -m llamafactory.cli export Week3/configs/merge_best_qwen.yaml

# ③ 抽查合并质量（对话行为应与 adapter 版一致）
.\.venv\Scripts\python.exe Week2/code/compare_finetune.py `
  --base models/Qwen2.5-3B-Instruct `
  --merged models/Qwen2.5-3B-week3-best-merged `
  --out Week3/deliverables/merge_sanity_check.md
```

## 收尾清单

- [ ] 《最优模型说明.md》填入最优超参与判定依据（⏳ 处）
- [ ] 周报 ⏳ 处全部填数：三组假设验证、盲测结论、OpenCompass 结论、验收自查表
- [ ] `collect_results.py --copy-logs` 确认 `deliverables/logs/` 完整（5 组）
- [ ] 交付物出 .docx 版本（如导师要求，同 Week1/2 惯例）
- [ ] git 提交 Week3（`saves/`、`models/`、`/data/`、`outputs/` 均已 gitignore）
