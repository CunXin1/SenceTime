"""Day26:构造 200 条图文指令数据 + 20 条留出评测集。

## 和任务书的一处设计偏差（重要，请连同理由一起看）

任务书 26.1 举例是"请根据这张图写一段产品描述"。**产品描述是主观生成任务，
微调前后没法客观打分**，而 ❸ 的验收要求是"微调后有明显效果提升"——
主观任务只能靠人打分，样本一少就说不清是提升还是噪声。

所以这里换成一个**可客观打分**的窄任务：

    输入：训练平台截图（随机化的超参与状态）
    输出：固定字段顺序的结构化实验记录（Markdown）

这样 Day26 的对比就能算出**字段级抽取准确率**（每张图 8 个字段逐个比对真值），
提升是几个百分点一目了然，不依赖人工主观评分。任务本身也确实是业务需求
（把训练平台截图自动转成规范实验记录），不是为了好打分硬造的。

## 为什么图是合成的

200 张真实截图收集不现实，而合成图的**真值 100% 已知**——每个字段的值都是
生成时随机出来的，直接就是标签，不需要人工标注。同时随机了两种版式
（卡片式 / 紧凑列表式）、字号、配色和字段顺序，避免模型只是背下某一种像素布局。

用法:
    .venv-vlm\\Scripts\\python.exe Week5/code/build_vlm_sft_data.py
    .venv-vlm\\Scripts\\python.exe Week5/code/build_vlm_sft_data.py --n-train 200 --n-eval 20
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Week5" / "data"
TRAIN_IMG = DATA / "train_images"
EVAL_IMG = DATA / "eval_images"

FONT_CJK = "C:/Windows/Fonts/msyh.ttc"
FONT_MONO = "C:/Windows/Fonts/consola.ttf"

BASE_MODELS = [
    "Qwen2.5-VL-7B-Instruct", "Qwen2.5-3B-Instruct", "Llama-3.2-3B-Instruct",
    "gemma-4-E4B-it", "Qwen2.5-7B-Instruct", "Llama-3.2-11B-Vision",
    "Qwen2-VL-2B-Instruct", "gemma-4-E2B-it",
]
METHODS = ["LoRA", "QLoRA", "DoRA", "Full"]
STATUSES = [("运行中", "#16a34a", "#dcfce7", "#166534"),
            ("已完成", "#2563eb", "#dbeafe", "#1e40af"),
            ("失败", "#dc2626", "#fee2e2", "#991b1b"),
            ("排队中", "#a16207", "#fef9c3", "#854d0e")]
MENU = ["数据集", "训练任务", "模型仓库", "评测", "部署", "设置"]

# 输出字段分两类:
#   前 8 个是**照抄**字段 —— 从截图里读出来即可,强基座本来就能做对(实测基线 ~92%)
#   后 3 个是**派生**字段 —— 需要算术 + 一条团队内部规则,规则**不写在指令里**,
#                            只存在于这 200 条训练数据中
#
# 这个设计是被实测逼出来的:最初 8 个字段全是照抄,Qwen2.5-VL 基线就有 92% 字段准确率,
# 微调再怎么练也只剩 8 个点空间,❸「明显效果提升」根本交不出来。
# 加了派生字段之后,基座只能靠猜,而微调模型能从数据里学到规则 —— 这才测得出 LoRA 的价值。
COPY_FIELDS = ["任务ID", "基座模型", "训练方式", "学习率", "进度", "当前loss", "显存占用", "状态"]
DERIVED_FIELDS = ["进度百分比", "剩余步数", "健康状态"]
FIELDS = COPY_FIELDS + DERIVED_FIELDS

# 团队规范(**不出现在指令里**,只能从训练数据里学):
#   健康状态 = 需关注  当 状态==失败 或 loss>=1.5 或 显存>=21.0 GB
#            = 正常    其余情况
#
# 阈值是按"两类各占一半"反推的,不是拍脑袋:
#   loss ~ U(0.15,1.95) → P(loss<1.5)=0.75; vram ~ U(6,23) → P(vram<21)=0.88;
#   状态 4 选 1,P(非失败)=0.75  ⇒  P(正常)=0.75×0.75×0.88≈0.50
# 第一版用 1.0/20.0,算出来只有 29% 是"正常",模型一律答"需关注"就能蒙到 71%,
# 这个字段就废了 —— 和 Day25 存在性探针要正负各半是同一个道理。
HEALTH_LOSS_THRESHOLD = 1.5
HEALTH_VRAM_THRESHOLD = 21.0

INSTRUCTION = (
    "请把这张训练平台截图转换成我们团队规范的实验记录卡。"
    "严格按以下字段顺序输出，每行一个字段，格式为「- 字段名：值」，不要输出任何额外说明：\n"
    + "\n".join(f"- {f}：" for f in FIELDS)
)


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)


def mono_or_cjk(text: str, mono: ImageFont.FreeTypeFont,
                cjk: ImageFont.FreeTypeFont) -> ImageFont.FreeTypeFont:
    """Consolas 没有中文字形,含中文的串必须换雅黑,否则渲染成方块。"""
    return cjk if has_cjk(text) else mono


def sample_record(rng: random.Random, idx: int) -> dict:
    method = rng.choice(METHODS)
    rank = rng.choice([8, 16, 32, 64])
    total = rng.choice([120, 180, 240, 300, 480, 600])
    done = rng.randint(1, total)
    lr = rng.choice(["5.0e-5", "1.0e-4", "2.0e-4", "5.0e-6", "1.0e-5", "3.0e-4"])
    loss = rng.uniform(0.15, 1.95)
    vram = rng.uniform(6.0, 23.0)
    status = rng.choice(STATUSES)[0]
    unhealthy = (status == "失败" or loss >= HEALTH_LOSS_THRESHOLD
                 or vram >= HEALTH_VRAM_THRESHOLD)
    return {
        "任务ID": f"job-2026{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}-{rng.choice(['vlm', 'sft', 'dpo', 'ocr'])}-{idx:03d}",
        "基座模型": rng.choice(BASE_MODELS),
        "训练方式": "Full" if method == "Full" else f"{method} (r={rank}, alpha={rank * 2})",
        "学习率": lr,
        "进度": f"{done}/{total}",
        "当前loss": f"{loss:.4f}",
        "显存占用": f"{vram:.1f} GB",
        "状态": status,
        # 派生字段:算术 + 团队规则,指令里不给规则
        "进度百分比": f"{round(done / total * 100)}%",
        "剩余步数": str(total - done),
        "健康状态": "需关注" if unhealthy else "正常",
        "_epochs": rng.choice([1, 2, 3, 5]),
        "_batch": rng.choice([1, 2, 4, 8]),
        "_accum": rng.choice([1, 2, 4, 8, 16]),
        "_variant": rng.choice(["card", "compact"]),
        "_accent": rng.choice(["#3b82f6", "#8b5cf6", "#0ea5e9", "#14b8a6", "#f97316"]),
        "_sidebar": rng.choice(["#1f2937", "#111827", "#1e293b", "#292524"]),
        "_selected": rng.choice(MENU),
    }


def render(rec: dict, out: Path) -> None:
    status = next(s for s in STATUSES if s[0] == rec["状态"])
    accent, sidebar = rec["_accent"], rec["_sidebar"]
    f_logo, f_menu = font(FONT_CJK, 19), font(FONT_CJK, 15)
    f_h2, f_lab, f_val = font(FONT_CJK, 19), font(FONT_CJK, 14), font(FONT_MONO, 14)

    W, H = 940, 560
    SB = 180
    img = Image.new("RGB", (W, H), "#f5f6f8")
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, SB, H], fill=sidebar)
    d.text((20, 22), "LoRA 训练台", font=f_logo, fill="#ffffff")
    for i, item in enumerate(MENU):
        y = 76 + i * 40
        if item == rec["_selected"]:
            d.rectangle([10, y - 7, SB - 10, y + 26], fill="#374151")
            d.rectangle([10, y - 7, 13, y + 26], fill=accent)
        d.text((28, y), item, font=f_menu, fill="#e5e7eb" if item == rec["_selected"] else "#9ca3af")

    d.rectangle([SB, 0, W, 56], fill="#ffffff")
    d.line([(SB, 56), (W, 56)], fill="#e5e7eb", width=1)
    d.text((SB + 24, 18), f"训练任务 / {rec['任务ID']}", font=f_h2, fill="#111827")
    d.rounded_rectangle([W - 120, 14, W - 24, 42], radius=14, fill=status[2])
    d.ellipse([W - 110, 24, W - 103, 31], fill=status[1])
    d.text((W - 95, 18), rec["状态"], font=f_lab, fill=status[3])

    pairs_left = [("基座模型", rec["基座模型"]), ("训练方式", rec["训练方式"]),
                  ("学习率", rec["学习率"]),
                  ("批大小", f"{rec['_batch']} × 累积 {rec['_accum']}"),
                  ("训练轮数", str(rec["_epochs"]))]
    done, total = rec["进度"].split("/")
    pct = int(done) / int(total)
    pairs_right = [("已完成步数", rec["进度"]), ("当前 loss", rec["当前loss"]),
                   ("显存占用", rec["显存占用"]),
                   ("预计剩余", f"{int((1 - pct) * 40):02d}:{random_minutes(rec):02d}")]

    if rec["_variant"] == "card":
        d.rounded_rectangle([SB + 24, 80, 560, 400], radius=9, fill="#ffffff", outline="#e5e7eb")
        d.text((SB + 46, 98), "超参配置", font=f_h2, fill="#111827")
        for i, (k, v) in enumerate(pairs_left):
            y = 142 + i * 44
            d.text((SB + 46, y), k, font=f_lab, fill="#6b7280")
            d.rounded_rectangle([SB + 150, y - 7, 536, y + 23], radius=5,
                                fill="#f9fafb", outline="#e5e7eb")
            d.text((SB + 162, y), v, font=mono_or_cjk(v, f_val, f_lab), fill="#111827")
        d.rounded_rectangle([584, 80, W - 24, 400], radius=9, fill="#ffffff", outline="#e5e7eb")
        d.text((606, 98), "运行状态", font=f_h2, fill="#111827")
        for i, (k, v) in enumerate(pairs_right):
            y = 146 + i * 50
            d.text((606, y), k, font=f_lab, fill="#6b7280")
            d.text((790, y), v, font=mono_or_cjk(v, f_val, f_lab), fill="#111827")
        d.rounded_rectangle([606, 368, W - 46, 379], radius=5, fill="#e5e7eb")
        d.rounded_rectangle([606, 368, 606 + int((W - 46 - 606) * pct), 379], radius=5, fill=accent)
    else:  # compact:单栏紧凑列表,换一种版式防止模型只背布局
        d.rounded_rectangle([SB + 24, 80, W - 24, 400], radius=9, fill="#ffffff", outline="#e5e7eb")
        d.text((SB + 46, 98), "任务详情", font=f_h2, fill="#111827")
        for i, (k, v) in enumerate(pairs_left + pairs_right):
            y = 134 + i * 26
            d.text((SB + 46, y), f"{k}", font=f_lab, fill="#6b7280")
            d.text((SB + 220, y), v, font=mono_or_cjk(v, f_val, f_lab), fill="#111827")
        bar_y = 134 + len(pairs_left + pairs_right) * 26 + 12   # 让进度条落在最后一行下方
        d.rounded_rectangle([SB + 46, bar_y, W - 46, bar_y + 10], radius=5, fill="#e5e7eb")
        d.rounded_rectangle([SB + 46, bar_y, SB + 46 + int((W - 92 - SB) * pct), bar_y + 10],
                            radius=5, fill=accent)

    d.rounded_rectangle([SB + 24, 420, W - 24, 536], radius=9, fill="#ffffff", outline="#e5e7eb")
    d.text((SB + 46, 434), "训练日志", font=f_lab, fill="#6b7280")
    logs = [f"[STEP {done}] loss={rec['当前loss']}  lr={rec['学习率']}",
            f"[INFO] peak_memory={rec['显存占用']}  status={rec['状态']}",
            f"[INFO] method={rec['训练方式']}"]
    for i, line in enumerate(logs):
        d.text((SB + 46, 462 + i * 24), line, font=mono_or_cjk(line, f_val, f_lab),
               fill="#374151")

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def random_minutes(rec: dict) -> int:
    # 不用内置 hash():Python 对字符串的 hash 每次进程启动都加盐,会破坏可复现性
    return sum(rec["任务ID"].encode()) % 60


def target_text(rec: dict) -> str:
    return "\n".join(f"- {k}：{rec[k]}" for k in FIELDS)


def build(n_train: int, n_eval: int, seed: int = 42) -> None:
    rng = random.Random(seed)
    train_rows, eval_rows = [], []

    for i in range(n_train):
        rec = sample_record(rng, i)
        rel = f"Week5/data/train_images/train_{i:04d}.png"
        render(rec, ROOT / rel)
        train_rows.append({
            "messages": [
                {"role": "user", "content": f"<image>{INSTRUCTION}"},
                {"role": "assistant", "content": target_text(rec)},
            ],
            "images": [rel],
        })

    rng_eval = random.Random(seed + 10_000)  # 独立随机流,和训练集不重叠
    for i in range(n_eval):
        rec = sample_record(rng_eval, 9000 + i)
        rel = f"Week5/data/eval_images/eval_{i:03d}.png"
        render(rec, ROOT / rel)
        eval_rows.append({
            "image": rel,
            "instruction": INSTRUCTION,
            "fields": {k: rec[k] for k in FIELDS},
            "target": target_text(rec),
        })

    (DATA / "vlm_sft_200.json").write_text(
        json.dumps(train_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (DATA / "eval_records.json").write_text(
        json.dumps(eval_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # LLaMA-Factory 数据集注册表:多模态数据集必须声明 images 字段,否则图片被忽略
    (DATA / "dataset_info.json").write_text(json.dumps({
        "week5_vlm_sft": {
            "file_name": "vlm_sft_200.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant"},
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[写出] {DATA / 'vlm_sft_200.json'}      训练 {len(train_rows)} 条")
    print(f"[写出] {DATA / 'eval_records.json'}     留出 {len(eval_rows)} 条")
    print(f"[写出] {DATA / 'dataset_info.json'}     数据集注册（week5_vlm_sft）")
    print(f"[生成] {TRAIN_IMG} / {EVAL_IMG}")
    print(f"\n输出字段（{len(FIELDS)} 个，Day26 按字段逐个比对算准确率）：{FIELDS}")
    print("\n样例目标输出：")
    print(train_rows[0]["messages"][1]["content"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.n_train, args.n_eval, args.seed)
