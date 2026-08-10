# Day21：周报与总结（收尾记录）

> 交付物：rewards 曲线图、《第4周_DPO偏好对齐报告.md》、最终 DPO 模型归档。

## 今天做了什么

1. `plot_rewards.py` 读各组 `trainer_state.json` 出 rewards 曲线（每组一张 + margins 叠加对比图）。
2. 整理 3 组 rewards 趋势、安全拒答率、对齐对比结论，撰写周报。
3. 归档最终模型 `models/Qwen2.5-3B-week4-dpo-merged`，更新《最优DPO模型说明.md》。

## 执行方式

```powershell
.\.venv\Scripts\python.exe Week4/code/plot_rewards.py
```

## 收尾清单

- [ ] 3 组训练完成，`DPO实验结果汇总.md` rewards 趋势判定
- [ ] rewards 曲线图 `rewards_*.png`
- [ ] 红线安全测试拒答率 ≥90%（硬指标）
- [ ] 对齐效果对比表 + 主观评分
- [ ] 最优 DPO 模型合并归档
- [ ] 周报 `第4周_DPO偏好对齐报告.md`
- [ ] （如导师要求）周报出 .docx 版本，同 Week1/2 惯例

## 下周衔接

最终 DPO 模型 `models/Qwen2.5-3B-week4-dpo-merged` 可作为后续量化部署（AWQ/GPTQ）或
进一步 RLHF/评测的输入。
