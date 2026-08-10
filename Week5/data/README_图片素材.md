# Week5 图片素材包说明

6 张图全部就绪。真值统一记录在 `images/ground_truth.json`（由 `prepare_images.py` 生成）。

| 文件 | 尺寸 | 类型 | 来源 | 主要考察点 |
|---|---|---|---|---|
| `01_table.png` | 1060×378 | 表格截图 | 自动生成（Week4 真实 DPO 数据） | 结构化 OCR、数字准确性 |
| `02_landscape.jpg` | 1536×1024 | 自然风景 | 实拍 | 描述能力、**计数幻觉**（画面恰好 2 人） |
| `03_logo.jpg` | 1280×900 | Logo | 实拍 | 世界知识关联（Nasdaq）、叠加文字 OCR |
| `04_signboard_jp.jpg` | 1707×1280 | 日文竖排书法招牌 | 实拍（明治神宫和歌板） | **跨语种 OCR**、竖排右起版式 |
| `05_ui.png` | 1180×720 | UI 界面 | 自动生成 | 空间关系、控件定位、数值抽取 |
| `06_chart.png` | 1080×600 | 业务图表 | 复用 Week4 产出 | 视觉→数值→推理链 |

## 关于 `04_signboard_jp.jpg`

原计划这一格是「手写公式」。实际放入的是明治神宫的和歌告示板——**印刷体日文竖排书法，
不是手写**。没有强行套用原分类，`ground_truth.json` 里如实标注为
「日文竖排书法招牌(替代原计划的手写公式)」。

保留它的理由：竖排 + 从右往左 + 书法字体 + 跨语种，OCR 难度不低于手写公式，
而且能直接拉开 Qwen2.5-VL（中日文强）和 gemma-4-E4B（CJK 弱）的差距，
是本周两模型对比里信息量最大的一张图。

如果要严格覆盖任务书的「手写公式」，补一张手写照片进来（建议写 `W' = W + (α/r)·BA`），
在 `prepare_images.py` 的 `GROUND_TRUTH` 里加一格 `07_handwriting.jpg` 即可，
`build_questions.py` 和 `build_hallu_probes.py` 会自动带上。

## `absent_objects` 为什么重要

Day25 的物体幻觉率 = 模型对 `absent_objects` 里的东西答「有」的比例。
这一项**只列 100% 确定画面里没有的东西**——有一个标错，整个幻觉率就不可信了。

校验：`.venv-vlm\Scripts\python.exe Week5/code/prepare_images.py --check`
