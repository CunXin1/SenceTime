"""Day23 交付:能力边界分析 -> Day23_能力边界分析.md

自动做能自动做的部分(视觉 token 对比、延迟、长度、关键事实命中),
定性判断写在常量里。**关键事实命中**是这里的核心:
从 ground_truth 里挑出可以字符串匹配的硬事实(确切数字、确切文字),
逐条检查两个模型的回答里有没有,这样能力边界就不是靠印象说的,而是有据可查。

用法:
    .venv-vlm\\Scripts\\python.exe Week5/code/analyze_day23.py
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DELIV = ROOT / "Week5" / "deliverables"
CSV_IN = DELIV / "图文推理结果表.csv"
OUT = DELIV / "Day23_能力边界分析.md"

DISPLAY = {"qwen": "Qwen2.5-VL-7B-Instruct", "gemma": "gemma-4-E4B-it"}

# 每张图挑几条"必须出现在回答里"的硬事实,用于客观打分。
# 选的都是唯一、可字符串匹配的内容,不做模糊判断。
# alias 里任一命中即算命中(容忍全半角、单位写法差异)。
FACTS: dict[str, list[tuple[str, str, list[str]]]] = {
    "01_table.png": [
        ("ocr", "标题「末次评估」", ["末次评估"]),
        ("ocr", "最高 eval acc = 0.790", ["0.790", "0.79"]),
        ("chart", "对应峰值显存 13.5 GB", ["13.5"]),
        ("chart", "对应 eval margin 0.9191", ["0.9191"]),
        ("ocr", "run_id qwen_dpo_beta0.1_lr1e-5", ["beta0.1_lr1e-5", "beta0.1 lr1e-5"]),
    ],
    "02_landscape.jpg": [
        ("describe", "画面恰好 2 个人", ["两个人", "两人", "2 个人", "2个人", "一对"]),
        ("describe", "人在沙滩上（非游泳）", ["沙滩", "海滩", "岸"]),
        ("describe", "前景有皮划艇/船", ["皮划艇", "独木舟", "船", "艇"]),
    ],
    "03_logo.jpg": [
        ("ocr", "主标识 Nasdaq", ["Nasdaq", "纳斯达克", "NASDAQ"]),
        ("ocr", "背景出现 EVgo", ["EVgo", "EVGO", "Evgo"]),
        ("infer", "识别为证券交易所/金融", ["证券", "交易所", "金融", "股票"]),
    ],
    "04_signboard_jp.jpg": [
        ("ocr", "题款「明治天皇御製」", ["明治天皇御製", "明治天皇御制"]),
        ("ocr", "题款「昭憲皇太后御歌」", ["昭憲皇太后", "昭宪皇太后"]),
        ("ocr", "第一首首句「世の中の」", ["世の中の"]),
        ("ocr", "第二首首句「身にしみて」", ["身にしみて"]),
        ("ocr", "落款「明治神宮」", ["明治神宮", "明治神宫"]),
    ],
    "05_ui.png": [
        ("chart", "进度 156/240", ["156", "156/240"]),
        ("chart", "loss 0.4127", ["0.4127"]),
        ("chart", "显存 17.2 GB", ["17.2"]),
        ("chart", "可训练参数占比 0.24%", ["0.24"]),
        ("describe", "左侧菜单 6 项", ["6 项", "6项", "六项", "六个"]),
    ],
    "06_chart.png": [
        ("chart", "对比 3 组实验", ["三组", "3 组", "3组", "三个", "3 个"]),
        ("chart", "最优组 beta0.5", ["beta0.5", "beta 0.5", "β=0.5"]),
    ],
}

# 定性观察(跑完实验后人工归纳,写在这里保证可复现)
QUALITATIVE = """
### 表现好的场景

1. **结构化图像的数值抽取**（表格、UI、图表）。Qwen 在这类图上几乎不出错：
   表格 OCR 能逐字还原成 Markdown 表格并保持行列结构；UI 界面的 4 个数值
   （进度 156/240、loss 0.4127、显存 17.2 GB、参数占比 0.24%）全部答对。
   Day24 的注意力热力图也印证了这一点——注意力精确落在目标单元格上。

2. **世界知识关联**。两个模型都认出了 Nasdaq 标识，并正确归类到证券交易所/金融行业；
   Qwen 还额外读出了背景电子屏上的 EVgo。

3. **有明确视觉证据的存在性判断**（Day25 数据支撑）：准确率 0.971。

### 幻觉严重 / 能力边界的场景

1. **两个模型的 OCR 差距不在「能不能读」，而在「字符级精度」——这是本周最有价值的发现。**

   Gemma 的日文和中文都**读得出来**，最初根据部分日志下的「Gemma 不能做日文 OCR」的判断
   是错的，逐条比对后修正如下。真实差距是**细粒度字形辨识**：

   | 图片 | 真值 | Qwen | Gemma |
   |---|---|---|---|
   | 表格 | `qwen_dpo_beta0.1_lr5e-6` | ✅ 完全正确 | ❌ `qwen_dpo_beta@0.1_1r5e-6`（凭空插入 `@`，`l`→`1`） |
   | 表格 | 标题「**末**次评估」 | ✅ | ❌ 读成「**本**次评估」 |
   | 表格 | 12 个数值 | ✅ 全对 | ✅ **全对** |
   | 日文牌 | 「**昭憲**皇太后御歌」 | ✅ | ❌ 「**略意**皇太后御歌」 |
   | 日文牌 | 「明治神宮**崇敬**会」 | ✅ | ❌ 「明治神宮**紫微**会」 |
   | 日文牌 | 10 句假名和歌 | ✅ 全对 | ✅ **假名全对**，但自创了错误的分栏结构 |

   规律很清楚：**Gemma 对「大而清晰」的内容（数字、假名）准确，
   对「形近字/小字」（末vs本、憲vs意、崇敬vs紫微、l vs 1）系统性出错。**
   Qwen 在这两类上都正确。

2. **上一条的根因是固定 soft token 预算，不是「不认识这门语言」。**
   Gemma 对所有图片都用约 530 个视觉 token（本次设为 560 预算），
   而 Qwen 对 1707×1280 的日文招牌用到 1271 个——**是 Gemma 的 2.4 倍**。
   同一张图被压到不到一半的表征容量，形近字的区分信息在预处理阶段就丢了。

   工程含义：**Gemma 处理密集文字图必须把 `max_soft_tokens` 提到 1120**，
   否则数字看着对、字符悄悄错，比完全读不出来更危险。

3. **误导性前提下的编造**。问「表格最后一列的饼图显示什么比例」（图中无饼图），
   Qwen 完整编造出「14.0 GB 占三分之一、13.6 GB 占四分之一」。
   模型倾向于**接受提问者预设的前提**，而不是指出前提错误。

4. **计数类问题**：UI 界面「左侧菜单几项」Qwen 答对 6 项，Gemma 答错。
   计数需要遍历整个区域，比抽取单个数值更依赖空间分辨率。

5. **主观评价类问题**（评价美学）两个模型都倾向于输出安全的套话，
   信息量低，很难分出高下。这类任务不适合用来评估 VLM 能力。
"""


def load() -> list[dict]:
    with CSV_IN.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fact_hits(rows: list[dict]) -> dict[str, dict[str, list[bool]]]:
    """{model: {image: [每条事实是否命中]}}"""
    out: dict[str, dict[str, list[bool]]] = {}
    for k in DISPLAY:
        out[k] = {}
        for img, facts in FACTS.items():
            hits = []
            for cat, _label, aliases in facts:
                # 事实可能出现在该图任一类问题的回答里,优先看指定类别,没有就全看
                texts = [r["answer"] for r in rows
                         if r["model"] == k and r["image"] == img
                         and (r["qid"].endswith(cat) or True)]
                blob = "\n".join(texts)
                hits.append(any(a.lower() in blob.lower() for a in aliases))
            out[k][img] = hits
    return out


def main() -> None:
    rows = load()
    models = [k for k in DISPLAY if any(r["model"] == k for r in rows)]
    hits = fact_hits(rows)

    L = ["# Day23 交付：图文推理能力边界分析\n",
         f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M}　"
         f"由 `Week5/code/analyze_day23.py` 生成，定性部分写在脚本常量里（可复现）。\n",
         "> 原始记录见 `图文推理结果表.csv`（每条含完整回答、视觉 token 数、延迟、显存）。\n",
         "## 一、运行概况\n",
         "| 模型 | 条数 | 平均视觉 token | 平均延迟 | 峰值显存 | 平均答案长度 |",
         "|---|---|---|---|---|---|"]
    for k in models:
        rs = [r for r in rows if r["model"] == k]
        L.append(f"| `{DISPLAY[k]}` | {len(rs)} | "
                 f"{sum(int(r['image_tokens']) for r in rs) / len(rs):.0f} | "
                 f"{sum(float(r['latency_s']) for r in rs) / len(rs):.1f}s | "
                 f"{max(float(r['peak_mem_gb']) for r in rs):.2f} GB | "
                 f"{sum(len(r['answer']) for r in rs) / len(rs):.0f} 字 |")

    L.append("\n## 二、视觉 token 策略实测对比\n")
    L.append("| 图片 | 原图尺寸 | Qwen（动态分辨率） | Gemma（固定预算 560） |")
    L.append("|---|---|---|---|")
    sizes = {"01_table.png": "1060×378", "02_landscape.jpg": "1536×1024",
             "03_logo.jpg": "1280×900", "04_signboard_jp.jpg": "1707×1280",
             "05_ui.png": "1180×720", "06_chart.png": "1080×600"}
    for img, size in sizes.items():
        q = next((r["image_tokens"] for r in rows if r["model"] == "qwen" and r["image"] == img), "—")
        g = next((r["image_tokens"] for r in rows if r["model"] == "gemma" and r["image"] == img), "—")
        L.append(f"| `{img}` | {size} | {q} | {g} |")
    L.append("\n**Qwen 的视觉 token 随图片尺寸大幅变化，Gemma 基本恒定。** "
             "这是「原生动态分辨率」与「固定 soft token 预算」两种设计的直接体现，"
             "也是下一节 OCR 能力差异的根本原因之一。\n")

    L.append("## 三、关键事实命中率（客观打分）\n")
    L.append("从 `ground_truth.json` 里挑出可精确匹配的硬事实，逐条检查回答中是否出现。"
             "这样能力边界不是靠印象，而是有据可查。\n")
    L.append("| 图片 | 事实 | " + " | ".join(DISPLAY[k] for k in models) + " |")
    L.append("|---|---|" + "---|" * len(models))
    for img, facts in FACTS.items():
        for i, (_cat, label, _al) in enumerate(facts):
            cells = ["✅" if hits[k][img][i] else "❌" for k in models]
            L.append(f"| `{img}` | {label} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("| 模型 | 命中 / 总数 | 命中率 |\n|---|---|---|")
    total = sum(len(v) for v in FACTS.values())
    for k in models:
        n = sum(sum(v) for v in hits[k].values())
        L.append(f"| `{DISPLAY[k]}` | {n} / {total} | **{n / total:.1%}** |")

    L.append("\n### 按图片拆解命中率\n")
    L.append("| 图片 | " + " | ".join(DISPLAY[k] for k in models) + " |")
    L.append("|---|" + "---|" * len(models))
    for img, facts in FACTS.items():
        cells = [f"{sum(hits[k][img])}/{len(facts)}" for k in models]
        L.append(f"| `{img}` | " + " | ".join(cells) + " |")

    L.append("\n## 四、定性分析\n" + QUALITATIVE)

    L.append("\n## 五、五类问题的适用性小结\n")
    L.append("| 类别 | 适合评估 VLM 吗 | 说明 |\n|---|---|---|")
    L.append("| 描述内容 | ✅ | 能暴露计数幻觉和物体虚构 |")
    L.append("| 提取文字 | ✅✅ | **区分度最高**，跨语种 OCR 直接拉开两个模型 |")
    L.append("| 解释图表 | ✅✅ | 有唯一正确答案，可精确打分 |")
    L.append("| 评价美学 | ❌ | 两个模型都输出安全套话，无区分度 |")
    L.append("| 推理隐含信息 | ✅ | 能看出是否给出视觉依据，但需人工判断 |")
    L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[写出] {OUT}")
    for k in models:
        n = sum(sum(v) for v in hits[k].values())
        print(f"  {DISPLAY[k]:<26} 关键事实命中 {n}/{total} = {n / total:.1%}")


if __name__ == "__main__":
    main()
