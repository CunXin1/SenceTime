"""Day23:生成 5 类问题 × N 张图 的问题矩阵 -> Week5/data/questions.json

5 类问题不是随便定的,每一类都在测一个独立能力维度:

  describe  视觉 grounding 基线 —— 能不能说清画面里到底有什么
  ocr       OCR + 结构保持    —— 能不能逐字读出文字并保留布局
  chart     视觉→数值→推理链  —— 能不能从图里取数并做比较
  aesthetic 主观生成          —— 最容易输出空话套话,测"没内容也硬说"的倾向
  infer     视觉常识推理      —— 能不能给出依据,而不是凭空编场景

同一类问题对不同图片要微调措辞(比如风景照问"图表趋势"就是无效问题),
所以这里用 模板 + 按图覆盖 的方式生成,而不是硬编码 25 条。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "Week5" / "data" / "images"
OUT = ROOT / "Week5" / "data" / "questions.json"

CATEGORIES = {
    "describe": "描述内容",
    "ocr": "提取文字",
    "chart": "解释图表",
    "aesthetic": "评价美学",
    "infer": "推理隐含信息",
}

# 通用模板
DEFAULT = {
    "describe": "详细描述这张图片的内容，包括你看到的所有主要元素及其位置关系。",
    "ocr": "逐字提取图中所有可见文字，尽量保持原有的排列布局。如果没有文字，请直接说明「图中没有文字」。",
    "chart": "这张图里是否有图表或数据？如果有，说明它反映的趋势并给出关键数值；如果没有，请直接说明「图中没有图表或数据」。",
    "aesthetic": "从构图、色彩、信息层次三个角度评价这张图，并指出一个具体可改进的地方。",
    "infer": "推断这张图最可能出现在什么场景/用途中，并逐条列出你的判断依据。",
}

# 按图覆盖:让问题真正贴合这张图要考察的点
OVERRIDES: dict[str, dict[str, str]] = {
    "01_table.png": {
        "ocr": "逐字提取这张表格中的所有文字和数字，用 Markdown 表格的形式输出，保持原有的行列结构。",
        "chart": "这张表里 eval acc 最高的是哪一行？它的 eval margin 和峰值显存分别是多少？请给出具体数值。",
        "infer": "这张表最可能来自什么工作场景？表里的 β 和 lr 两个变量说明作者在做什么实验？逐条给依据。",
    },
    "02_landscape.jpg": {
        "describe": "详细描述这张照片的内容。请明确说出画面中可数物体的数量（例如有几个人、几棵树、几栋建筑）。",
        "chart": "这张图里是否有图表或数据？如果没有，请直接说明「图中没有图表或数据」，不要编造。",
        "infer": "推断这张照片的拍摄季节、大致时间（上午/正午/傍晚）和地理环境类型，并逐条给出你的视觉依据。",
    },
    "03_logo.jpg": {
        "describe": "描述这张图的构成：前景的标识长什么样（图形和文字分别是什么、什么颜色），背景是什么。",
        "ocr": "提取图中所有可见文字，包括前景的主标识和背景屏幕上的文字。背景文字如果看不清，请说明「看不清」，不要猜。",
        "chart": "这张图里是否有图表或数据？如果没有，请直接说明「图中没有图表或数据」，不要编造。",
        "infer": "这是哪个品牌或组织的标识？它属于什么行业？背景拍的是什么地方？说明你的识别依据。如果不确定，请明确说「不确定」。",
    },
    "04_signboard_jp.jpg": {
        "describe": "描述这张照片：画面里有什么物体、告示板长什么样、周围环境是什么。",
        "ocr": "这块告示板上的文字是竖排、从右往左书写的。请按正确的阅读顺序逐字提取全部文字，"
               "并标出哪几行是标题（作者署名）、哪几行是正文。看不清的字用 [?] 标注，不要猜。",
        "chart": "这张图里是否有图表或数据？如果没有，请直接说明「图中没有图表或数据」，不要编造。",
        "infer": "这块牌子最可能立在什么场所？上面写的是什么体裁的文字？分别出自谁？逐条给出依据。",
    },
    "05_ui.png": {
        "describe": "详细描述这个界面的布局结构：有哪些区域、每个区域放了什么、左侧菜单共有几项、当前选中哪一项。",
        "ocr": "逐字提取界面上的所有文字，按「侧边栏 / 顶栏 / 左卡片 / 右卡片 / 日志区 / 按钮」分区列出。",
        "chart": "界面上的进度是多少？当前 loss 和显存占用分别是多少？可训练参数占比是多少？请给出具体数值。",
        "infer": "这是什么类型的产品界面？使用者正在做什么任务？从日志里能推断出哪些训练配置？逐条给依据。",
    },
    "06_chart.png": {
        "chart": "这张图对比了几组实验？各条曲线的走势分别是什么？哪一组的 margins 末端值最大？给出具体数值。",
        "ocr": "提取图中所有文字，包括标题、坐标轴标签和图例。",
        "infer": "这张图想证明什么结论？作者做这组实验的目的是什么？逐条给依据。",
    },
}

# 5 张核心图(交付表要求 25 条);06 是加分项
CORE_IMAGES = ["01_table.png", "02_landscape.jpg", "03_logo.jpg",
               "04_signboard_jp.jpg", "05_ui.png"]
BONUS_IMAGES = ["06_chart.png"]


def build() -> list[dict]:
    rows = []
    for img in CORE_IMAGES + BONUS_IMAGES:
        for cat, cat_cn in CATEGORIES.items():
            rows.append({
                "qid": f"{img.split('_')[0]}-{cat}",
                "image": img,
                "image_exists": (IMG_DIR / img).exists(),
                "category": cat,
                "category_cn": cat_cn,
                "is_core": img in CORE_IMAGES,
                "question": OVERRIDES.get(img, {}).get(cat, DEFAULT[cat]),
            })
    return rows


if __name__ == "__main__":
    rows = build()
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    core = sum(r["is_core"] for r in rows)
    ready = sum(r["image_exists"] for r in rows)
    print(f"[写出] {OUT}")
    print(f"  总计 {len(rows)} 条(核心 {core} 条 = 5图×5类,加分 {len(rows) - core} 条)")
    print(f"  其中图片已就绪的 {ready} 条,缺图 {len(rows) - ready} 条")
    missing = sorted({r['image'] for r in rows if not r['image_exists']})
    if missing:
        print(f"  缺图: {missing} —— 见 Week5/data/README_图片素材.md")
