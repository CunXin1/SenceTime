# Day14：自动化评估与人工盲测打分（操作指南）

> 交付物：各模型答卷（`deliverables/eval_answers/`）、5 维评分卡、盲测材料、
> 雷达图 + 最优模型标记（`radar_*.png`、`盲测得分汇总.md`）。

## 评估对象

- 6 个模型：Qwen 基座 + 5 个 LoRA 实验产物（adapter 热挂载，不合并）。
- 题集：`Week3/data/eval_questions.json` —— 20 道固定高难度题
 （数学 7 / 推理 7 / 代码 6），每题带打分用参考要点。

## 第一步：批量作答（`eval_harness.py`）

```powershell
# 训练全部结束、GPU 空闲后运行（约 6 模型 × 20 题，贪心解码可复现）
.\.venv\Scripts\python.exe Week3/code/eval_harness.py
# 冒烟：先 2 题验证挂载与输出格式
.\.venv\Scripts\python.exe Week3/code/eval_harness.py --max-questions 2
```

设计要点：
- **基座只加载一次，PEFT 热挂/卸载 adapter**——不做 5 次合并导出，省磁盘 ~30GB。
- **贪心解码（do_sample=False）**：任何人重跑得到完全相同的答案，打分可复审。
- 输出双格式：`answers_<run_id>.json`（给脚本）+ `<run_id>.md`（给人看）。

## 第二步：生成盲测材料（`make_scorecard.py`）

```powershell
# 全部 6 个模型入围（数量可控，无需再筛）
.\.venv\Scripts\python.exe Week3/code/make_scorecard.py
```

产出三件套：
| 文件 | 给谁 |
|---|---|
| `盲测答卷.md` | 打分人（模型名已匿名为 Model-A/B/C…，按题目分组便于横向对比） |
| `盲测打分记录.csv` | 打分人（Excel 直接打开，5 维度空白列） |
| `盲测映射表.md` / `盲测映射.json` | **保密**，打分结束前不发 |

匿名映射用固定种子洗牌，可复现；6 模型 × 20 题 × 5 维 = 600 格，
单人约 1.5~2 小时可完成。

## 第三步：邀请打分（需要人工环节）

把《人工评分卡.md》+《盲测答卷.md》+《盲测打分记录.csv》发给同事/导师；
维度与锚点详见评分卡（准确 30 / 完整 25 / 逻辑 20 / 安全 15 / 格式 10）。
建议至少 1 位非本人打分者；两人打分时同题取平均。

## 第四步：回收出图（`make_radar.py`）

```powershell
.\.venv\Scripts\python.exe Week3/code/make_radar.py
```

自动完成：加权总分计算 → `radar_overview.png`（全模型叠加）+ 每模型单图 →
`盲测得分汇总.md`（★ 标记最优）。CSV 允许部分填写（空格自动跳过），
可以边打边预览。
