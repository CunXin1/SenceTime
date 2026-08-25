"""
build_tech_report.py — Week8 Day44
把 8 个分章 Markdown 拼成一份完整技术报告：统一参考文献编号、生成目录、统计字数、转 .docx。
Assemble the 8 per-chapter Markdown files into one technical report.

★ 为什么分章写、最后再拼
    八章由不同人/不同时间写成，如果一开始就往一个大文件里塞，任何两处并行编辑都会打架。
    分章 + 拼装脚本把「写」和「排」解耦：写的人只关心自己那一章的内容与本章引用，
    编号、目录、字数、格式统一由脚本一次性处理，且**可重复执行**——
    改完某一章重跑一次就行，不需要手工维护全局编号。
    Per-chapter authoring + a deterministic assembler: no merge conflicts,
    and global numbering is recomputed on every build.

★ 参考文献为什么要重编号
    每章末尾各自维护 `### 本章引用`，编号从 [1] 开始。直接拼起来会出现四个 [1]。
    本脚本把各章的局部编号映射到全局编号，替换正文里的角标，最后汇成一个「参考文献」章。
    **同一篇文献在多章被引时会合并成一个全局编号**（按标题去重），不重复列出。
    Local [1..n] per chapter → global numbering, deduplicated by title.

★ 替换角标时必须跳过代码块
    正文里的 `[1]` 是引用，但代码块里的 `attn[0, head, ...]`、`plt.subplots()[1]`
    形状完全一样。脚本按 ``` 围栏切分，**只在非代码段做替换**——
    否则会把代码改坏，而且这种改坏在渲染出来的文档里极难发现。
    Skip fenced code blocks: `[1]` inside code is an index, not a citation.

★ 字数口径
    「≥6000 字」按**中文字符数**统计（CJK 统一表意文字 + 常用标点），
    不含代码块、表格分隔线、图片路径、HTML 注释——这些不是"写"出来的正文。
    脚本会把两种口径都打出来，避免用一个虚高的数字自欺。
    Word count excludes code fences, table rules and image paths.

用法 / Usage（仓库根目录）:
    .venv/Scripts/python.exe Week8/scripts/build_tech_report.py
    .venv/Scripts/python.exe Week8/scripts/build_tech_report.py --no-docx
产物 / Output:
    Week8/reports/技术报告_Qwen2.5-3B全链路实践.md
    Week8/reports/技术报告_Qwen2.5-3B全链路实践.docx
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "Week8" / "reports"
OUT_MD = REPORTS / "技术报告_Qwen2.5-3B全链路实践.md"

# 章节顺序即文件顺序。缺章时脚本会明确报出来，而不是悄悄少一章。
CHAPTERS = [
    "ch1_环境与架构.md",
    "ch2_数据工程.md",
    "ch3_SFT优化实验.md",
    "ch4_DPO偏好对齐.md",
    "ch5_多模态与Agent实践.md",
    "ch6_部署与量化.md",
    "ch7_全链路自动化.md",
    "ch8_总结与展望.md",
]

TITLE = "Qwen2.5-3B 全链路实践技术报告"
SUBTITLE = "从数据清洗到量化服务：单卡 24GB 上的八周工程记录"

CITE_HEADING = "### 本章引用"
_CITE_LINE = re.compile(r"^\[(\d+)\]\s*(.+?)\s*$")
_PAREN = re.compile(r"[（(][^）)]*[）)]")
_NONWORD = re.compile(r"[^0-9a-z一-鿿]+")
_INTEXT = re.compile(r"\[(\d+)\]")
_FENCE = re.compile(r"^```")
# 中文字符：CJK 统一表意文字 + 中文标点
_CJK = re.compile(r"[一-鿿　-〿＀-￯]")


def split_body_and_cites(text: str) -> tuple[str, list[tuple[int, str]]]:
    """把一章拆成正文与本章引用列表。没有引用小节时返回空表。"""
    idx = text.find(CITE_HEADING)
    if idx < 0:
        return text.rstrip(), []
    body = text[:idx].rstrip()
    cites = []
    for line in text[idx + len(CITE_HEADING):].splitlines():
        m = _CITE_LINE.match(line.strip())
        if m:
            cites.append((int(m.group(1)), m.group(2)))
    return body, cites


def cite_key(entry: str) -> str:
    """把一条文献压成用于去重的键：**归一化后的标题**。

    ★ 为什么不能直接拿整条字符串比（2026-08-25 实测）
      各章由不同人/不同时间写成，同一篇论文的写法并不一致：
          ch2  "Hu E J, Shen Y, Wallis P, et al. LoRA: Low-Rank Adaptation
                of Large Language Models. ICLR, 2022."
          ch5  "Hu E., et al. *LoRA: Low-Rank Adaptation of Large Language
                Models*. ICLR 2022."
      字符串比对下这是两条，于是同一篇 LoRA 出现两个全局编号。第一版就是这样，
      26 条文献里有 5 组是同一篇的不同写法（LoRA / LlamaFactory / vLLM /
      C-Eval / AWQ）。参考文献表里同一篇列两次，是很显眼的硬伤。
      Exact-string dedup fails: chapters cite the same paper in different styles.

    ★ 归一化取"最长的点分段"作为标题
      文献条目的结构基本都是 `作者. 标题. 出处, 年份.`，三段里标题几乎总是最长的
      （作者被缩写、出处是会议简称）。取最长段再去掉 markdown 强调符、括号补充
      （"(vLLM)"这类）、大小写和所有标点，剩下的就是可比的标题骨架。
      作者列表的写法差异（"Hu E J, Shen Y, Wallis P, et al" vs "Hu E., et al"）
      正好被排除在键之外——它们恰恰是最容易不一致的部分。
      Key = normalized longest dot-segment (the title), so author-style
      differences do not split one paper into two entries.

    保留的是**首次出现**的完整写法，因此各章的引用风格以最先出现的那一章为准。
    """
    s = entry.replace("*", "").replace("_", " ")
    s = _PAREN.sub(" ", s)
    seg = max((p for p in s.split(".")), key=lambda p: len(p.strip()), default=s)
    return _NONWORD.sub("", seg.lower())


def renumber(body: str, mapping: dict[int, int]) -> str:
    """把正文里的局部角标换成全局角标，**跳过代码块**（理由见文件头）。"""
    out, in_code = [], False
    for line in body.splitlines():
        if _FENCE.match(line.strip()):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        out.append(_INTEXT.sub(
            lambda m: f"[{mapping[int(m.group(1))]}]" if int(m.group(1)) in mapping else m.group(0),
            line))
    return "\n".join(out)


def count_chars(text: str) -> tuple[int, int]:
    """返回 (中文字符数, 全部非空白字符数)。剔除代码块 / 表格分隔线 / 图片路径。"""
    kept, in_code = [], False
    for line in text.splitlines():
        s = line.strip()
        if _FENCE.match(s):
            in_code = not in_code
            continue
        if in_code:
            continue
        if re.match(r"^\|[\s\-:|]+\|$", s):     # 表格的分隔行
            continue
        s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)   # 图片
        s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)  # 链接保留文字
        kept.append(s)
    body = "\n".join(kept)
    return len(_CJK.findall(body)), len(re.sub(r"\s", "", body))


def build_toc(bodies: list[str]) -> str:
    """从各章的 ## / ### 标题生成目录。只到二级，再深目录会长得没法看。"""
    lines = []
    for b in bodies:
        in_code = False
        for line in b.splitlines():
            if _FENCE.match(line.strip()):
                in_code = not in_code
                continue
            if in_code:
                continue
            if line.startswith("## "):
                lines.append(f"- **{line[3:].strip()}**")
            elif line.startswith("### "):
                lines.append(f"  - {line[4:].strip()}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-docx", action="store_true", help="只出 Markdown，不转 docx")
    args = ap.parse_args()

    missing = [c for c in CHAPTERS if not (REPORTS / c).exists()]
    if missing:
        sys.exit("[FAIL] 缺少章节文件：\n  " + "\n  ".join(missing))

    bodies: list[str] = []
    # 全局文献表：按**归一化后的标题**去重（见 cite_key），保持首次出现的顺序
    global_cites: list[str] = []
    cite_index: dict[str, int] = {}          # 归一化键 → 全局编号（1-based）
    per_chapter_counts: list[tuple[str, int, int]] = []

    for name in CHAPTERS:
        raw = (REPORTS / name).read_text(encoding="utf-8")
        body, cites = split_body_and_cites(raw)
        mapping = {}
        for local_n, title in cites:
            key = cite_key(title)
            if key in cite_index:
                mapping[local_n] = cite_index[key]
            else:
                global_cites.append(title)
                cite_index[key] = len(global_cites)
                mapping[local_n] = len(global_cites)
        body = renumber(body, mapping)
        bodies.append(body)
        cjk, allc = count_chars(body)
        per_chapter_counts.append((name, cjk, allc))

    toc = build_toc(bodies)
    total_cjk = sum(c for _, c, _ in per_chapter_counts)
    total_all = sum(a for _, _, a in per_chapter_counts)

    parts = [
        f"# {TITLE}",
        f"> **{SUBTITLE}**",
        "",
        "> 硬件：Windows 11 + NVIDIA RTX 4090 (24GB) · 基座：Qwen2.5-3B-Instruct · "
        "框架：LLaMA-Factory 0.9.6.dev0 / vLLM 0.27.1",
        "",
        "---",
        "",
        "## 目录",
        "",
        toc,
        "",
        "---",
        "",
    ]
    parts.extend(b + "\n\n---\n" for b in bodies)

    if global_cites:
        parts.append("## 参考文献\n")
        parts.extend(f"[{i}] {t}  " for i, t in enumerate(global_cites, 1))

    OUT_MD.write_text("\n".join(parts) + "\n", encoding="utf-8")

    print("=== 分章字数（中文字符 / 非空白字符）===")
    for name, cjk, allc in per_chapter_counts:
        print(f"  {name:<32} {cjk:>6} / {allc:>6}")
    print(f"  {'合计':<32} {total_cjk:>6} / {total_all:>6}")
    print(f"  参考文献 {len(global_cites)} 条")
    if total_cjk < 6000:
        print(f"  ⚠️  中文字符数 {total_cjk} < 6000，未达要求")
    else:
        print(f"  ✅ 中文字符数 {total_cjk} ≥ 6000")
    print(f"[ok] {OUT_MD}")

    if not args.no_docx:
        # 复用 Week4 写好的转换器（支持标题/表格/图片/加粗，中文字体微软雅黑）
        conv = ROOT / "Week4" / "code" / "md_to_docx.py"
        py = ROOT / ".venv" / "Scripts" / "python.exe"
        r = subprocess.run([str(py if py.exists() else sys.executable), str(conv), str(OUT_MD)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(r.stdout.strip() or r.stderr.strip())


if __name__ == "__main__":
    main()
