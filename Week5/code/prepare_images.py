"""Day22:生成/整理 5+1 张测试图片,并写出 ground_truth.json。

为什么要 ground_truth:Day25 要算幻觉率,必须先知道每张图里"客观上有什么、没有什么"。
凡是模型说出 absent_objects 里的东西,就判定为物体幻觉。

自动生成(真值 100% 可控,最适合量化幻觉):
    01_table.png   表格截图  —— 用 Week4 真实 DPO 实验数据渲染
    05_ui.png      UI 界面   —— 合成的训练平台控制台
    06_chart.png   业务图表  —— 直接复用 Week4/deliverables/rewards_overview.png

需人工放入(真实照片才有意义,脚本只做占位与校验):
    02_landscape.jpg  自然风景(要求:画面里有可数物体,便于测计数幻觉)
    03_logo.png       Logo 截图(测知识关联)
    04_handwriting.jpg 手写公式照片(最容易 OCR 幻觉)

用法:
    .venv\\Scripts\\python.exe Week5/code/prepare_images.py
"""
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "Week5" / "data" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

FONT_CJK = "C:/Windows/Fonts/msyh.ttc"      # 微软雅黑,渲染中文
FONT_MONO = "C:/Windows/Fonts/consola.ttf"  # Consolas,渲染数字


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)


def pick(text: str, mono: ImageFont.FreeTypeFont, cjk: ImageFont.FreeTypeFont):
    """Consolas 没有中文字形,含中文的串必须换微软雅黑,否则渲染成方块。"""
    return cjk if has_cjk(text) else mono


# --------------------------------------------------------------------------
# 01 表格截图:Week4 DPO 实验的"末次评估 + 训练成本"表
# --------------------------------------------------------------------------
TABLE_HEADER = ["run_id", "eval acc", "eval margin", "train loss", "耗时", "峰值显存"]
TABLE_ROWS = [
    ["qwen_dpo_beta0.1_lr5e-6", "0.774", "0.5132", "0.5790", "15m39s", "14.0 GB"],
    ["qwen_dpo_beta0.5_lr5e-6", "0.758", "1.2466", "0.4755", "14m46s", "13.6 GB"],
    ["qwen_dpo_beta0.1_lr1e-5", "0.790", "0.9191", "0.4816", "14m38s", "13.5 GB"],
]
COL_W = [300, 130, 150, 140, 120, 140]


def make_table() -> Path:
    f_title = font(FONT_CJK, 26)
    f_head = font(FONT_CJK, 19)
    f_cell = font(FONT_MONO, 19)
    pad, row_h, top = 40, 52, 110
    w = pad * 2 + sum(COL_W)
    h = top + row_h * (len(TABLE_ROWS) + 1) + pad + 20

    img = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(img)
    d.text((pad, 34), "Week4 DPO 实验：末次评估与训练成本", font=f_title, fill="#1a1a1a")
    d.line([(pad, 78), (w - pad, 78)], fill="#d0d0d0", width=2)

    # 表头
    x = pad
    d.rectangle([pad, top, w - pad, top + row_h], fill="#f2f4f7")
    for text, cw in zip(TABLE_HEADER, COL_W):
        d.text((x + 14, top + 16), text, font=f_head, fill="#333333")
        x += cw
    # 数据行
    for i, row in enumerate(TABLE_ROWS):
        y = top + row_h * (i + 1)
        if i % 2 == 1:
            d.rectangle([pad, y, w - pad, y + row_h], fill="#fafbfc")
        x = pad
        for text, cw in zip(row, COL_W):
            d.text((x + 14, y + 16), text, font=f_cell, fill="#1a1a1a")
            x += cw
    # 网格线
    for i in range(len(TABLE_ROWS) + 2):
        y = top + row_h * i
        d.line([(pad, y), (w - pad, y)], fill="#e0e0e0", width=1)
    x = pad
    for cw in COL_W:
        d.line([(x, top), (x, top + row_h * (len(TABLE_ROWS) + 1))], fill="#e0e0e0", width=1)
        x += cw
    d.line([(w - pad, top), (w - pad, top + row_h * (len(TABLE_ROWS) + 1))], fill="#e0e0e0", width=1)

    out = IMG_DIR / "01_table.png"
    img.save(out)
    return out


# --------------------------------------------------------------------------
# 05 UI 界面:合成的模型训练控制台(虚构产品名,避免冒用真实品牌)
# --------------------------------------------------------------------------
UI_MENU = ["数据集", "训练任务", "模型仓库", "评测", "部署", "设置"]
UI_FIELDS = [
    ("基座模型", "Qwen2.5-VL-7B-Instruct"),
    ("训练方式", "LoRA (rank=16, alpha=32)"),
    ("学习率", "1.0e-4"),
    ("批大小", "1  ×  梯度累积 8"),
    ("训练轮数", "3"),
]
UI_STATS = [("已完成步数", "156 / 240"), ("当前 loss", "0.4127"), ("显存占用", "17.2 GB"), ("预计剩余", "08:42")]


def make_ui() -> Path:
    w, h = 1180, 720
    f_logo = font(FONT_CJK, 21)
    f_menu = font(FONT_CJK, 17)
    f_h2 = font(FONT_CJK, 22)
    f_lab = font(FONT_CJK, 16)
    f_val = font(FONT_MONO, 16)
    f_small = font(FONT_CJK, 14)

    img = Image.new("RGB", (w, h), "#f5f6f8")
    d = ImageDraw.Draw(img)

    # 左侧边栏
    d.rectangle([0, 0, 210, h], fill="#1f2937")
    d.text((24, 26), "LoRA 训练台", font=f_logo, fill="#ffffff")
    for i, item in enumerate(UI_MENU):
        y = 90 + i * 46
        if item == "训练任务":
            d.rectangle([12, y - 8, 198, y + 30], fill="#374151")
            d.rectangle([12, y - 8, 15, y + 30], fill="#60a5fa")
        d.text((32, y), item, font=f_menu, fill="#e5e7eb" if item == "训练任务" else "#9ca3af")

    # 顶栏
    d.rectangle([210, 0, w, 62], fill="#ffffff")
    d.line([(210, 62), (w, 62)], fill="#e5e7eb", width=1)
    d.text((240, 20), "训练任务 / job-20260805-vlm-sft", font=f_h2, fill="#111827")
    d.rounded_rectangle([w - 150, 16, w - 30, 46], radius=15, fill="#dcfce7")
    d.ellipse([w - 138, 27, w - 130, 35], fill="#16a34a")
    d.text((w - 122, 22), "运行中", font=f_lab, fill="#166534")

    # 左卡片:超参配置
    d.rounded_rectangle([240, 92, 690, 420], radius=10, fill="#ffffff", outline="#e5e7eb")
    d.text((266, 114), "超参配置", font=f_h2, fill="#111827")
    for i, (k, v) in enumerate(UI_FIELDS):
        y = 164 + i * 48
        d.text((266, y), k, font=f_lab, fill="#6b7280")
        d.rounded_rectangle([380, y - 8, 664, y + 26], radius=6, fill="#f9fafb", outline="#e5e7eb")
        d.text((392, y), v, font=pick(v, f_val, f_lab), fill="#111827")

    # 右卡片:运行状态
    d.rounded_rectangle([714, 92, 1150, 420], radius=10, fill="#ffffff", outline="#e5e7eb")
    d.text((740, 114), "运行状态", font=f_h2, fill="#111827")
    for i, (k, v) in enumerate(UI_STATS):
        y = 168 + i * 56
        d.text((740, y), k, font=f_lab, fill="#6b7280")
        d.text((980, y - 2), v, font=f_val, fill="#111827")
    # 进度条 156/240 = 65%
    d.rounded_rectangle([740, 388, 1124, 400], radius=6, fill="#e5e7eb")
    d.rounded_rectangle([740, 388, 740 + int(384 * 0.65), 400], radius=6, fill="#3b82f6")

    # 底部日志卡片
    d.rounded_rectangle([240, 444, 1150, 676], radius=10, fill="#ffffff", outline="#e5e7eb")
    d.text((266, 464), "训练日志", font=f_h2, fill="#111827")
    logs = [
        "[INFO] loading vision tower ... frozen (freeze_vision_tower=True)",
        "[INFO] trainable params: 20,185,088 / 8,394,342,400 (0.24%)",
        "[STEP 150] loss=0.4310  lr=8.21e-05  grad_norm=0.62",
        "[STEP 156] loss=0.4127  lr=8.05e-05  grad_norm=0.58",
    ]
    for i, line in enumerate(logs):
        d.text((266, 508 + i * 30), line, font=f_val, fill="#374151")
    d.rounded_rectangle([266, 630, 366, 662], radius=6, fill="#3b82f6")
    d.text((292, 638), "暂停", font=f_lab, fill="#ffffff")
    d.rounded_rectangle([382, 630, 482, 662], radius=6, fill="#ffffff", outline="#d1d5db")
    d.text((408, 638), "停止", font=f_lab, fill="#dc2626")
    d.text((266, 690 - 6), "", font=f_small, fill="#9ca3af")

    out = IMG_DIR / "05_ui.png"
    img.save(out)
    return out


# --------------------------------------------------------------------------
# 06 业务图表:复用 Week4 真实产出
# --------------------------------------------------------------------------
def copy_chart() -> Path | None:
    src = ROOT / "Week4" / "deliverables" / "rewards_overview.png"
    if not src.exists():
        print(f"[跳过] 未找到 {src}")
        return None
    out = IMG_DIR / "06_chart.png"
    shutil.copy(src, out)
    return out


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------
GROUND_TRUTH = {
    "01_table.png": {
        "type": "表格截图",
        "source": "自动生成(Week4 真实 DPO 数据)",
        "description": "白底表格截图,标题「Week4 DPO 实验：末次评估与训练成本」,6 列 3 行数据",
        "exact_text": {
            "title": "Week4 DPO 实验：末次评估与训练成本",
            "header": TABLE_HEADER,
            "rows": TABLE_ROWS,
        },
        "key_facts": [
            "共 3 行数据行 + 1 行表头",
            "eval acc 最高的是 qwen_dpo_beta0.1_lr1e-5,值为 0.790",
            "eval margin 最高的是 qwen_dpo_beta0.5_lr5e-6,值为 1.2466",
            "峰值显存三行分别是 14.0 GB / 13.6 GB / 13.5 GB",
        ],
        "present_objects": ["表格", "文字", "数字"],
        "absent_objects": ["折线图", "柱状图", "饼图", "图片", "人物", "logo", "按钮", "红色高亮"],
    },
    "02_landscape.jpg": {
        "type": "自然风景",
        "source": "人工放入,实拍照片(1536×1024)",
        "description": "海湾沙滩场景。前景是一排翻扣在岸边的皮划艇(黄绿、白底黑纹、蓝、橙等)"
                       "和一块绿色防水布;中景沙滩水边站着 2 个人,左边一人戴蓝色帽子、"
                       "穿浅灰上衣和深色长裤,右边一人穿酒红色上衣和黑色短裤,均赤脚;"
                       "远景是平静的海面和覆盖绿色植被、露出岩石的山丘岬角。晴天,略有薄雾。",
        "exact_text": {},
        "key_facts": [
            "画面中有且仅有 2 个人",
            "两人都站在沙滩上、没有人在水里游泳",
            "前景至少有 4 条翻扣的皮划艇/独木舟",
            "左边那个人戴着蓝色帽子",
            "海面平静,没有明显的浪花或白色浪头",
        ],
        # absent 只列"100% 确定画面里没有"的东西 —— Day25 的物体幻觉率完全依赖这一项的正确性
        "present_objects": ["海水", "沙滩", "人", "皮划艇", "山丘", "植被", "岩石", "帽子"],
        "absent_objects": ["猫", "狗", "汽车", "帆船", "太阳伞", "鸟", "雪", "文字招牌"],
    },
    "03_logo.jpg": {
        "type": "Logo",
        "source": "人工放入,实拍照片(1280×900)",
        "description": "Nasdaq 品牌标识叠加在一张压暗的城市夜景照片上。左侧是青蓝色折角「N」"
                       "图形标,右侧是白色无衬线的「Nasdaq」字样。背景是纳斯达克 MarketSite "
                       "大楼的弧形电子屏,屏上隐约可见「Welcome to the Nasdaq Family」、"
                       "「EVGO Nasdaq Listed」、大字「EVgo」以及底部另一个小号 Nasdaq 标识。",
        "exact_text": {
            "主标识": "Nasdaq",
            "背景屏幕": ["Welcome to the Nasdaq Family", "EVGO Nasdaq Listed",
                         "EVgo", "Nasdaq"],
        },
        "key_facts": [
            "品牌是 Nasdaq(纳斯达克证券交易所)",
            "图形标是青蓝色(#00a4d9 一类)的折角 N,文字标是白色",
            "背景是纳斯达克 MarketSite 大楼的弧形电子广告屏",
            "屏上出现的另一个公司名是 EVgo",
        ],
        "present_objects": ["Nasdaq标志", "英文文字", "高层建筑", "电子广告屏"],
        "absent_objects": ["手写文字", "表格", "二维码", "动物", "海洋", "山", "中文文字", "数学公式"],
    },
    "04_signboard_jp.jpg": {
        # 原计划这一格是"手写公式",实际放入的是日文竖排书法招牌。
        # 没有强行套用原分类:如实标注类型,并保留它作为 OCR 难度样本的价值 ——
        # 竖排 + 右起 + 书法字体 + 跨语种,比手写公式更能拉开中/美两个模型的差距。
        "type": "日文竖排书法招牌(替代原计划的手写公式)",
        "source": "人工放入,实拍照片(1707×1280),明治神宫和歌告示板",
        "description": "树林中一块白底、金色边框的大型告示板,由黑色金属立柱支撑,"
                       "板下是绿色灌木和草地。板上是竖排、从右往左书写的日文书法字体和歌两首。",
        "exact_text": {
            "阅读顺序": "竖排，从右往左",
            "第一首": {
                "题": "明治天皇御製",
                "歌": "世の中の / 事ある時に / あひぬとも / おのがつとめむ / ことな忘れそ",
            },
            "第二首": {
                "题": "昭憲皇太后御歌",
                "歌": "身にしみて / うれしきものは / まこともて / 人のつげたる / ことばなりけり",
            },
            "落款": ["明治神宮", "明治神宮崇敬会"],
        },
        "key_facts": [
            "板上共有两首和歌,分别标注「明治天皇御製」和「昭憲皇太后御歌」",
            "文字是竖排、从右往左阅读",
            "落款是「明治神宮」和「明治神宮崇敬会」",
            "告示板由黑色金属立柱支撑,背景是树林",
            "全部文字均为日文,没有中文简体字也没有英文句子",
        ],
        "present_objects": ["告示板", "日文文字", "树木", "灌木", "金属立柱", "草地"],
        "absent_objects": ["人物", "动物", "汽车", "图表", "数学公式", "二维码", "红色印章"],
    },
    "05_ui.png": {
        "type": "UI 界面",
        "source": "自动生成(虚构产品「LoRA 训练台」)",
        "description": "深色左侧边栏 + 浅色主区的训练管理控制台,含超参配置卡片、运行状态卡片、训练日志卡片",
        "exact_text": {
            "sidebar_title": "LoRA 训练台",
            "menu": UI_MENU,
            "page_title": "训练任务 / job-20260805-vlm-sft",
            "status_badge": "运行中",
            "fields": dict(UI_FIELDS),
            "stats": dict(UI_STATS),
            "buttons": ["暂停", "停止"],
        },
        "key_facts": [
            "左侧菜单共 6 项,当前选中「训练任务」",
            "进度为 156 / 240 步,进度条约 65%",
            "当前 loss 为 0.4127,显存占用 17.2 GB",
            "底部有 2 个按钮:暂停(蓝色实心)、停止(白底红字)",
            "日志显示 vision tower 已冻结,可训练参数占比 0.24%",
        ],
        "present_objects": ["侧边栏", "卡片", "进度条", "按钮", "状态徽章", "日志文本"],
        "absent_objects": ["折线图", "柱状图", "头像", "搜索框", "弹窗", "表格", "图片缩略图"],
    },
    "06_chart.png": {
        "type": "业务图表(加分项)",
        "source": "Week4/deliverables/rewards_overview.png",
        "description": "Week4 三组 DPO 实验的 rewards 趋势对比图",
        "exact_text": {},
        "key_facts": [
            "对比 3 组实验:beta0.1_lr5e-6 / beta0.5_lr5e-6 / beta0.1_lr1e-5",
            "chosen 奖励上升、rejected 奖励下降、margins 上升、accuracies 上升",
            "beta0.5_lr5e-6 的末端 margins 最大(1.9788)",
        ],
        "present_objects": ["折线图", "坐标轴", "图例"],
        "absent_objects": ["照片", "人物", "表格", "手写文字"],
    },
}

PENDING_README = """# Week5 图片素材包说明

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

校验：`.venv-vlm\\Scripts\\python.exe Week5/code/prepare_images.py --check`
"""


def check() -> int:
    print(f"{'文件':<22} {'状态':<8} 尺寸")
    missing = 0
    for name in GROUND_TRUTH:
        p = IMG_DIR / name
        if p.exists():
            with Image.open(p) as im:
                print(f"{name:<22} {'✅ 就绪':<8} {im.size[0]}x{im.size[1]}")
        else:
            print(f"{name:<22} {'❌ 缺失':<8} —— 见 README_图片素材.md")
            missing += 1
    # 检查 ground_truth 占位符
    raw = json.dumps(GROUND_TRUTH, ensure_ascii=False)
    holes = raw.count("【")
    if holes:
        print(f"\n⚠️  ground_truth.json 还有 {holes} 处【】占位符待填写(影响 Day25 幻觉判定)")
    return missing


if __name__ == "__main__":
    import sys

    if "--check" not in sys.argv:
        print("[生成] 01_table.png  ->", make_table())
        print("[生成] 05_ui.png     ->", make_ui())
        print("[复制] 06_chart.png  ->", copy_chart())

        (IMG_DIR / "ground_truth.json").write_text(
            json.dumps(GROUND_TRUTH, ensure_ascii=False, indent=2), encoding="utf-8")
        (IMG_DIR.parent.parent / "data" / "README_图片素材.md").write_text(
            PENDING_README, encoding="utf-8")
        print("[写出] ground_truth.json + README_图片素材.md\n")

    sys.exit(0 if check() == 0 else 0)  # 缺图不算失败,只提示
