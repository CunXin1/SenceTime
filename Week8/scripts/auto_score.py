"""
auto_score.py — Week8 Day41
把 Week3 的 5 维**人工**评分卡，实现成一套确定性、可解释的**自动代理指标**。
A deterministic, explainable proxy for Week3's 5-dimension human score card.

★ 首先要说清楚它不是什么（这一条比任何实现细节都重要）
    人工打分考察的是"这个回答对不对、好不好"，那需要真正读懂内容。本脚本做不到，
    它做的是**表面特征匹配**：答案里有没有出现正确的最终值、有没有推理链的形状、
    有没有烂尾/复读。因此它是人工评分的 **proxy（代理指标）**，不是替代品。
    它的正当用途只有两个：
      ① 回归护栏——同一套题、同一套规则，模型改版后分数掉了能立刻发现；
      ② 流水线自动化——Day41 要求"无人值守跑完 data→train→eval→deploy"，
        这一环不能等人来读 100 份答卷。
    它**不能**用来下"A 模型比 B 模型好"的结论，除非差距远大于下文实测的偏差。
    This is a PROXY for human grading (surface-feature matching), not a
    replacement. Use it as a regression guard and for unattended pipelines.

★ 为什么是"规则"而不是"LLM-as-judge"
    LLM 裁判分更高，但它①要占一张卡（Day41 明确要求礼让 GPU，其它 agent 在训练）
    ②不确定（同一份答卷两次跑分不同，回归护栏就废了）③无法解释（掉分说不出原因）。
    规则法反过来：零显存、完全可复现、每一分都能追到具体哪条规则。
    这三点正好是"流水线里的自动评估"最需要的。代价是天花板低——它读不懂内容。
    Rule-based: zero VRAM, bit-reproducible, every point traceable to a rule.
    An LLM judge would be more accurate but non-deterministic and GPU-hungry.

★ 两档匹配：为什么要单独维护 Week8/configs/eval.yaml
    见 eval.yaml 文件头。一句话：reference 字段是散文，里面的中间量会污染
    "准确性"的判定。精确档用蒸馏过的 final/support/anti 三段式；没有 answer_key
    的题自动退回**通用档**（从 reference 里机械抽数字与关键词），精度低但零配置。
    每题用了哪一档会写进逐题明细，CSV 里也有一列统计。
    Tier-1 uses the distilled answer key; tier-2 falls back to mechanical
    extraction from `reference` for question sets without a key.

★ 分数区间为什么取 [0,5] 而不是任务书字面的"0-5"里的 0 起
    Week3 人工评分卡的锚点是 **1~5**（1=完全错误，不是 0）。为了能和
    Week3/deliverables/盲测得分汇总.md 直接逐维对比，本脚本正常路径输出 1~5，
    只有"答案为空"这一种情况给 0。加权公式与人工卡完全一致：
    0.30 准确 + 0.25 完整 + 0.20 逻辑 + 0.15 安全 + 0.10 格式。

用法 / Usage（仓库根目录）:
    # 给一份答卷打分
    .venv/Scripts/python.exe Week8/scripts/auto_score.py \
        --answers Week3/deliverables/eval_answers/answers_qwen_base.json

    # ★ 用 Week3 的 100 条真实人工打分记录验证这套规则（不需要 GPU）
    .venv/Scripts/python.exe Week8/scripts/auto_score.py --validate

被 step3_eval.py 作为库导入：`from auto_score import AutoScorer`
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "Week3" / "data" / "eval_questions.json"
CONFIG = ROOT / "Week8" / "configs" / "eval.yaml"
WEEK3_ANSWERS = ROOT / "Week3" / "deliverables" / "eval_answers"
HUMAN_CSV = ROOT / "Week3" / "deliverables" / "盲测打分记录.csv"
HUMAN_MAP = ROOT / "Week3" / "deliverables" / "盲测映射.json"

DIMS = ["accuracy", "completeness", "logic", "safety", "format"]
DIM_CN = {"accuracy": "准确性", "completeness": "完整性", "logic": "逻辑性",
          "safety": "安全性", "format": "格式"}

# 兜底权重：eval.yaml 不存在时用，数值与 Week3 人工评分卡一致。
DEFAULT_WEIGHTS = {"accuracy": 0.30, "completeness": 0.25, "logic": 0.20,
                   "safety": 0.15, "format": 0.10}

# ---------------------------------------------------------------------------
# 结构信号词表。分三类是因为它们在"逻辑性"里权重不同：
#   step  = 显式的步骤标号，最强信号（作者明确在分步）
#   conn  = 推理连接词，中等信号（有推理链但没编号）
#   —— 公式行（含 '=' 的行）单独统计，数学/代码题里它比连接词更有信息量。
STEP_PAT = re.compile(
    r"(?:^|\n)\s*(?:\d{1,2}\s*[.)、．]|[①②③④⑤⑥⑦⑧⑨⑩]|第\s*[一二三四五六七八九十\d]+\s*[步趟]|步骤\s*\d)")
CONN_WORDS = ["因为", "所以", "因此", "首先", "其次", "然后", "接下来", "接着",
              "由此", "于是", "综上", "代入", "设", "根据", "假设", "可得",
              "得到", "推出", "最后", "即", "故", "由于", "考虑"]
CONN_PAT = re.compile("|".join(CONN_WORDS))

# 结论句窗口：用来做"自相矛盾"检测——同一题在多个结论句里给出不同的数。
CONCLUSION_PAT = re.compile(
    r"(?:所以|因此|综上|故|答案(?:是|为)?|结果(?:是|为)?|最终|答[:：])[^。\n]{0,30}")

HEDGE_PAT = re.compile(r"可能|大约|约为|不确定|也许|估计|近似|假设.{0,4}正确|需要?进一步")


# ---------------------------------------------------------------------------
# 数值归一化
# ---------------------------------------------------------------------------
def extract_numbers(text: str) -> set[float]:
    """把一段回答里所有"能当成数看"的东西抽成浮点集合。

    ★ 这是"准确性"这一维能不能用的关键。同一个答案 2.4 小时，模型可能写成
      "2.4"、"2.4小时"、"2 小时 24 分钟"、"12/5"、"\\frac{12}{5}"；
      如果只做字符串匹配，后三种全判错，"准确性"就失去意义了。
      所以这里把复合形式统一折算成同一个浮点数再比。
    Normalize every numeric surface form (decimal, fraction, LaTeX \\frac,
    "X小时Y分", percentage) into one float set so答案的写法差异不影响判分。

    有意做成**过度生成**（同一处可能贡献多个候选值）：本函数只服务于
    "目标值在不在里面"的存在性判断，宁可多给候选也不要漏。副作用是长答案里
    偶然出现的数字可能造成误命中——这是已知局限，写进了自动评分说明.md。
    """
    t = (text or "")
    # 全角数字/负号 → 半角；千分位逗号在后面单独处理
    t = t.translate(str.maketrans("０１２３４５６７８９．", "0123456789."))
    t = t.replace("−", "-").replace("–", "-")
    out: set[float] = set()

    def add(v: float) -> None:
        if abs(v) < 1e12:
            out.add(round(v, 6))

    # 1) "X小时Y分" / "X时Y分钟" → X + Y/60（放最前面，composite 优先）
    for h, m in re.findall(r"(\d+(?:\.\d+)?)\s*(?:个)?\s*(?:小时|时)\s*(\d+(?:\.\d+)?)\s*分", t):
        add(float(h) + float(m) / 60.0)
    # "X分Y秒" 同理（本题集用不到，但换题集时会用到）
    for m_, s in re.findall(r"(\d+(?:\.\d+)?)\s*分钟?\s*(\d+(?:\.\d+)?)\s*秒", t):
        add(float(m_) + float(s) / 60.0)

    # 2) LaTeX \frac{a}{b} / \dfrac{a}{b}，前置负号要跟着走
    for sign, a, b in re.findall(r"(-?)\s*\\[dt]?frac\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}", t):
        try:
            v = float(a) / float(b)
            add(-v if sign == "-" else v)
        except ZeroDivisionError:
            pass

    # 3) 普通分数 a/b。注意别把日期/版本号当分数——这里只接受纯数字两侧。
    for a, b in re.findall(r"(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", t):
        try:
            add(float(a) / float(b))
        except ZeroDivisionError:
            pass

    # 4) 百分数：13.89% 既记 0.1389 也记 13.89（模型可能两种口径都在写）
    for p in re.findall(r"(-?\d+(?:\.\d+)?)\s*%", t):
        add(float(p) / 100.0)

    # 5) 普通数字，含千分位逗号（11,576.25）
    for raw in re.findall(r"-?\d[\d,]*(?:\.\d+)?", t):
        try:
            add(float(raw.replace(",", "")))
        except ValueError:
            pass
    return out


def num_hit(target: float, pool: set[float], tol: float) -> bool:
    """目标值是否落在答案抽出的数值池里（相对误差 tol，接近 0 时退化为绝对误差）。"""
    scale = max(abs(target), 1.0)
    return any(abs(v - target) <= tol * scale for v in pool)


# ---------------------------------------------------------------------------
# 匹配原语
# ---------------------------------------------------------------------------
def squeeze(text: str) -> str:
    """压掉全部空白，供 `any` 的字面子串匹配使用。

    为什么要压：'[2, 2, 2]' 和 '[2,2,2]' 是同一个答案，模型的空格习惯不该影响判分。
    Collapse whitespace so spacing habits don't affect literal matching.
    """
    return re.sub(r"\s+", "", text or "")


class TextView:
    """一段文本的三种表示（原文 / 压掉空白 / 数值池），只算一次反复用。"""

    __slots__ = ("raw", "sq", "pool")

    def __init__(self, raw: str):
        self.raw = raw or ""
        self.sq = squeeze(self.raw)
        self.pool = extract_numbers(self.raw)


def conclusion_text(answer: str) -> str:
    """截出"结论区"：带结论标记的句子 + 末尾 200 字。

    ★ 这是把准确性从 4.42 拉回 4.03（人工 3.99）的那一刀（2026-08-25 实测）
      第一版在**全文**里找答案要点，于是这样的回答被判成答对：
        reason-01 模型写"如果乙说的是真话，那么…（矛盾）… 所以，甲说的是真话。"
        —— 全文里出现了"乙说的是真话"，但那是被**否定掉的假设**，结论恰恰是错的。
        reason-04 模型写"2 天后就是星期五 …… 所以 100 天之后是星期日。"
        —— "星期五"出现在举例里，结论 星期日 是错的。
        reason-02 模型写"照片里的这个人是说话者父亲的儿子 …… 是兄弟关系。"
        —— "儿子"是在复述题干。
      这三题人工分都在 1.6~3.2，自动分却给到 3.9~4.7。根因是同一个：
      **推理过程里必然会出现候选答案，只有结论区里的才算数**。
      所以带 `scope: conclusion` 的要点只在这段文本里匹配。
      Reasoning traces necessarily mention candidate answers (as hypotheses or
      as quotes); only what survives into the conclusion counts as the answer.
    """
    a = (answer or "").strip()
    if not a:
        return ""
    # 结论标记句：从标记词起到本句句号止（不跨句），所以举例/假设不会被带进来
    parts = [m.group(0) for m in re.finditer(
        r"(?:所以|因此|综上所述|综上|故|答案(?:是|为)?|结论(?:是|为)?[:：]?|"
        r"最终(?:答案)?|因而|可见|由此可知)[^。\n]{0,60}[。\n]?", a)]
    # ★ 尾窗只取 100 字，不是 200（2026-08-25 第二轮实测）
    #   200 字的尾窗会把推理过程的最后两三句一起圈进来，reason-01 的
    #   "如果乙说的是真话…"、reason-06 题干复述的"5 分钟"就是这么漏进结论区的，
    #   两题分别虚高 2.45 / 1.84 分。缩到 100 字后，尾窗基本只剩最后一句。
    #   A 200-char tail still swallowed hypothesis sentences; 100 keeps the last
    #   sentence only.
    parts.append(a[-100:])
    text = "\n".join(parts)
    # 再删掉假设/举例从句：它们里面的候选答案是被讨论的对象，不是结论
    return re.sub(r"(?:如果|假如|假设|若|例如|比如)[^。\n]*", "", text)


def match_item(item: dict, views: dict[str, TextView], tol: float) -> bool:
    """判断一个要点（num / any / any_re 三选一）是否被答案命中。

    `scope` 缺省为 full（全文）；写成 conclusion 时只在结论区里找，见上。
    """
    v = views.get(item.get("scope", "full"), views["full"])
    if "num" in item:
        return num_hit(float(item["num"]), v.pool, tol)
    for lit in item.get("any", []) or []:
        if squeeze(lit) in v.sq:
            return True
    for pat in item.get("any_re", []) or []:
        if re.search(pat, v.raw, re.IGNORECASE | re.MULTILINE):
            return True
    return False


# ---------------------------------------------------------------------------
# 结构特征（逻辑性 / 完整性 / 格式共用，只算一次）
# ---------------------------------------------------------------------------
def structure_features(answer: str) -> dict:
    """一次性抽出所有表面结构特征，避免各维度重复扫描文本。"""
    a = answer or ""
    lines = [ln.strip() for ln in a.splitlines()]
    fences = a.count("```")
    # 复读检测：同一条"有内容的行"出现 >=3 次，基本可以断定是循环生成。
    #
    # ★ 必须先剥掉行首的列表序号再比（2026-08-25 实测踩坑）
    #   reason-05 有一份答卷是这样的：
    #       1. 农夫带着羊过河。 2. 农夫返回原地。 3. 农夫带着狼过河。 …
    #       … 49. 农夫带着狼过河。 50.（在这里被 512 token 截断）
    #   ——农夫在原地来回摆渡了 50 趟，是彻底的循环生成，人工只给 1.85 分。
    #   但逐行原样比对完全抓不到：每行的序号都不一样，没有任何两行是相同字符串。
    #   剥掉 "N." 之后 "农夫返回原地。" 重复 25 次，一抓一个准。
    #   Strip list numbering first: a repeating plan has distinct line prefixes
    #   but identical bodies.
    counts: dict[str, int] = {}
    for ln in lines:
        body = re.sub(r"^\s*(?:\d{1,3}\s*[.)、．]|[①②③④⑤⑥⑦⑧⑨⑩]|[-*+]\s)\s*", "", ln)
        if len(body) >= 6:
            counts[body] = counts.get(body, 0) + 1
    max_line_repeat = max(counts.values(), default=0)
    # 更隐蔽的复读：某个 30 字窗口重复出现（跨行复读，行内容不完全相同也能抓到）
    win_repeat = 0
    if len(a) >= 120:
        seen: dict[str, int] = {}
        for i in range(0, len(a) - 30, 15):
            w = squeeze(a[i:i + 30])
            if len(w) >= 20:
                seen[w] = seen.get(w, 0) + 1
        win_repeat = max(seen.values(), default=0)

    # 代码块里的缩进是否崩了。3B 模型很爱把 4 空格缩进吐成 1 空格，
    # 贴出来的 Python 直接语法错——人工打分时这会明确扣"格式"分，
    # 但字符级的"有没有代码块"检查完全看不见它。
    # 3B models often emit 1-space indentation; the snippet won't even parse.
    broken_indent = False
    for blk in re.findall(r"```[^\n]*\n(.*?)```", a, re.S):
        if re.search(r"\b(def|if|for|while|class)\b", blk) and \
                re.search(r"(?m)^ (?! )\S", blk):
            broken_indent = True

    return {
        "len": len(a),
        "broken_indent": broken_indent,
        "n_steps": len(STEP_PAT.findall(a)),
        "n_conn": len(set(CONN_PAT.findall(a))),          # 去重：连接词种类比总数更能反映推理链
        "n_eq": sum(1 for ln in lines if "=" in ln),
        "n_fence": fences,
        "fence_balanced": fences % 2 == 0,
        "has_code": fences >= 2 or bool(re.search(r"\bdef\s+\w+|\bSELECT\b", a, re.I)),
        "has_md": bool(re.search(r"(^|\n)\s*(#{1,6}\s|[-*+]\s|\d+\.\s)|\*\*.+?\*\*", a)),
        "n_tests": len(re.findall(r"\bassert\b|\bprint\s*\(|测试用例|test_case|# *测试", a, re.I)),
        "n_bullets": len(re.findall(r"(?:^|\n)\s*(?:[-*+]\s|\d{1,2}\s*[.)、]|[①②③④⑤⑥⑦⑧⑨⑩])", a)),
        "max_line_repeat": max_line_repeat,
        "win_repeat": win_repeat,
        # 烂尾判定：正常回答收尾一定是句末标点、代码块围栏或右括号。
        # 512 token 上限下截断很常见，这正是"格式"这一维最该抓的东西。
        "truncated": not bool(re.search(r"[。！？!?\.\)）\]｝}`:：]\s*$", a.strip())) if a.strip() else True,
    }


# ---------------------------------------------------------------------------
class AutoScorer:
    """一次加载配置，对任意 (question_id, answer) 打 5 维分。"""

    def __init__(self, config_path: Path = CONFIG, questions_path: Path = QUESTIONS):
        self.cfg = {}
        if config_path.exists():
            self.cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        else:
            print(f"[warn] 找不到 {config_path}，全部题目退回**通用档**（精度更低）。")
        self.weights = self.cfg.get("weights") or DEFAULT_WEIGHTS
        self.tol = float(self.cfg.get("number_tolerance", 0.002))
        self.key = self.cfg.get("answer_key") or {}
        sp = self.cfg.get("safety_patterns") or {}
        self.harmful = [re.compile(p) for p in sp.get("harmful", [])]
        self.refusal = [re.compile(p, re.MULTILINE) for p in sp.get("refusal", [])]
        qs = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
        self.q = {x["id"]: x for x in qs}

    # -- 通用档：没有 answer_key 时的机械抽取 ------------------------------
    def _generic_facets(self, qid: str) -> tuple[list[dict], list[dict]]:
        """从 reference 里机械抽 final/support。

        ★ 取舍：把 reference 里出现、而**题面里没出现**的数字当作"答案相关量"。
          题面出现过的数（如鸡兔同笼的 35、94）是给定条件，模型抄一遍不代表答对，
          剔掉它们能显著降低误判。剩下的仍然混着中间量，所以通用档只能是
          "覆盖率"而非"对错"——这是它精度低于精确档的根本原因。
          Drop numbers that already appear in the question stem; what remains
          still mixes intermediates, hence tier-2 measures coverage, not truth.
        """
        q = self.q.get(qid, {})
        ref, stem = q.get("reference", ""), q.get("question", "")
        stem_nums = extract_numbers(stem)
        ref_nums = [v for v in extract_numbers(ref)
                    if not num_hit(v, stem_nums, self.tol)]
        # 数字太多时只留"最不像中间量"的几个：绝对值最大 + 非整数优先
        ref_nums.sort(key=lambda v: (float(v).is_integer(), -abs(v)))
        final = [{"num": v} for v in ref_nums[:2]]
        # 关键词：reference 里的 CJK/英文词块，长度 >=2，去掉太常见的连接词
        toks = [w for w in re.findall(r"[一-鿿]{2,6}|[A-Za-z_][A-Za-z_0-9]{2,}", ref)
                if not CONN_PAT.fullmatch(w)]
        support = [{"any": [w]} for w in dict.fromkeys(toks)][:4]
        if not final and support:                 # 纯文字题（无数字答案）
            final, support = support[:1], support[1:]
        return final, support

    # -- 各维度 ------------------------------------------------------------
    def _accuracy(self, spec, views, det) -> float:
        finals = spec.get("final") or []
        if not finals:
            det["acc_note"] = "无 final 要点，准确性不可判，记 3.0（中性）"
            return 3.0
        hits = [match_item(it, views, self.tol) for it in finals]
        rate = sum(hits) / len(hits)
        score = 1.0 + 4.0 * rate                  # 全中 5，全不中 1（对齐人工锚点）
        # anti 一律是正则，且只在**结论区**匹配：推导中间出现过的错值不算数,
        # 只有被当成最终答案端出来的才扣分。
        concl = views["conclusion"].raw or views["full"].raw
        anti = [p for p in (spec.get("anti") or []) if re.search(p, concl)]
        if anti:
            # 命中已知典型错答：即使正确值也在文中（常见于"先算错再改对"），
            # 也要扣——人工卡里这属于"答案部分正确"。
            score -= 1.5
        det["acc_final_hits"] = f"{sum(hits)}/{len(hits)}"
        if anti:
            det["acc_anti"] = anti
        return max(0.0, min(5.0, score))

    def _completeness(self, spec, views, f, det) -> float:
        sups = spec.get("support") or []
        sup_rate = (sum(match_item(it, views, self.tol) for it in sups)
                    / len(sups)) if sups else 1.0
        reqs = spec.get("requires") or []
        req_ok: list[float] = []
        for r in reqs:
            if r == "process":
                # "请给出完整推理过程" 这类硬要求：至少要有推理链的形状
                req_ok.append(1.0 if (f["n_steps"] + f["n_conn"] + f["n_eq"]) >= 4 else 0.0)
            elif r == "formula":
                req_ok.append(1.0 if f["n_eq"] >= 1 else 0.0)
            elif r == "enumerate":
                req_ok.append(1.0 if f["n_bullets"] >= 3 else 0.0)
            elif r == "code":
                req_ok.append(1.0 if f["has_code"] else 0.0)
            elif r.startswith("tests>="):
                need = int(r.split(">=")[1])
                req_ok.append(min(1.0, f["n_tests"] / need))
            else:
                req_ok.append(1.0)
        req_rate = statistics.fmean(req_ok) if req_ok else (1.0 if f["len"] > 150 else 0.5)
        score = 1.0 + 4.0 * (0.5 * sup_rate + 0.5 * req_rate)
        if f["len"] < 60:                          # 一句话回答，谈不上完整
            score = min(score, 2.5)
        if f["truncated"]:                         # 写到一半被截断 = 内容残缺
            score -= 0.5
        det["comp_support"] = f"{round(sup_rate, 2)}"
        det["comp_requires"] = f"{round(req_rate, 2)}"
        return max(0.0, min(5.0, score))

    def _logic(self, answer, f, acc_rate, det) -> float:
        """逻辑性 = 推理链的**形状** + 明显破绽扣分。

        ★ 为什么改成"高起点 + 扣分"，而不是"按信号量线性加分"（2026-08-25 实测）
          第一版是 1 + 4×加权信号量，实测全模型均值 2.54，人工是 3.97，
          系统性低了 1.43 分——这是五维里最大的偏差，光靠调系数救不回来。
          根因是**建模方式错了**：人工打分卡的锚点是"5=完整自洽 / 3=有明显跳步或
          一处矛盾 / 2=多处矛盾"，人是**默认给高分再扣**的；而线性加分模型默认
          给低分再加，于是所有语气平实、不爱用连接词的回答（尤其是代码题）都被
          冤枉。改成同样的"默认自洽、见到破绽才扣"结构后，均值回到 3.9 附近。
          The human rubric grades top-down (assume coherent, deduct on defects);
          a bottom-up "count the connectives" model systematically under-scores.
        """
        has_chain = (f["n_conn"] >= 2 or f["n_steps"] >= 2 or f["n_eq"] >= 2)
        strong = (f["n_conn"] >= 4 or f["n_steps"] >= 3
                  or (f["n_conn"] >= 2 and f["n_eq"] >= 3))
        score = 3.0 + (1.0 if has_chain else 0.0) + (0.7 if strong else 0.0)
        if f["n_steps"] >= 3:
            score += 0.3                            # 显式分步：读者不用自己补链条
        if not has_chain and f["len"] > 200:
            score -= 0.8                            # 长篇大论却看不出推理结构

        # 自相矛盾：多个结论句给出**不同**的数值（"答案是 42 … 所以答案是 40"）
        vals: set[float] = set()
        for w in CONCLUSION_PAT.findall(answer):
            vals |= extract_numbers(w)
        if len(vals) >= 3:
            score -= 0.8
            det["logic_contradiction"] = f"结论句中出现 {len(vals)} 个不同数值"
        if f["max_line_repeat"] >= 3 or f["win_repeat"] >= 4:
            score -= 1.0                            # 复读 = 推理链断了在原地打转
            det["logic_repeat"] = True
        if acc_rate == 0.0:
            # 推理过程再漂亮，结论错了就说明链条中间断过——人工也是这么扣的
            score -= 0.8
            det["logic_wrong_conclusion"] = True
        elif acc_rate < 1.0:
            score -= 0.3
        det["logic_signals"] = f"conn={f['n_conn']} step={f['n_steps']} eq={f['n_eq']}"
        return max(0.0, min(5.0, score))

    def _safety(self, answer, f, acc_rate, det) -> float:
        """安全性 = 有害内容 + 无端拒答 + 幻觉代理。

        ★ 幻觉代理是这一维里唯一"有信息量"的规则，也是最弱的一条：
          最终答案错、且语气笃定（有结论句）、且没有任何不确定性措辞——
          这在人工卡里对应"编造数字污染核心答案"。它必然误伤"算错但态度诚恳"
          的回答，所以扣分给得很轻（-0.6 / -1.0），不会把某个模型一杆子打死。
          The hallucination proxy (confidently-stated wrong answer) is the
          weakest rule here; penalties are kept small on purpose.
        """
        score = 5.0
        if any(p.search(answer) for p in self.harmful):
            det["safety_harmful"] = True
            return 1.0
        if any(p.search(answer.strip()) for p in self.refusal):
            det["safety_refusal"] = True
            score = 2.0
        assertive = bool(CONCLUSION_PAT.search(answer)) and not HEDGE_PAT.search(answer)
        if acc_rate == 0.0:
            score -= 1.4 if assertive else 0.9
            det["safety_hallucination"] = ("笃定地给出了错误结论" if assertive
                                           else "结论错误（有不确定性措辞，轻扣）")
        elif acc_rate < 1.0:
            score -= 0.7
            det["safety_hallucination"] = "结论部分正确"
        if f["max_line_repeat"] >= 3:
            score -= 0.4
        return max(0.0, min(5.0, score))

    def _format(self, spec, f, det) -> float:
        score, notes = 5.0, []
        if f["len"] == 0:
            return 0.0
        if f["truncated"]:
            score -= 1.5
            notes.append("烂尾/被截断")
        if not f["fence_balanced"]:
            score -= 1.0
            notes.append("代码块围栏未闭合")
        if f["len"] > 4000:
            score -= 1.0
            notes.append("过长")
        elif f["len"] < 30:
            score -= 1.5
            notes.append("过短")
        if f["max_line_repeat"] >= 3 or f["win_repeat"] >= 4:
            score -= 1.0
            notes.append("复读")
        if "code" in (spec.get("requires") or []) and f["n_fence"] < 2:
            score -= 0.8
            notes.append("代码没放进代码块")
        if f["broken_indent"]:
            score -= 0.6
            notes.append("代码缩进异常（1 空格，贴出来跑不了）")
        if f["len"] > 300 and not f["has_md"]:
            score -= 0.5
            notes.append("长回答无任何分段/列表结构")
        det["format_notes"] = notes
        return max(0.0, min(5.0, score))

    # -- 对外接口 ----------------------------------------------------------
    def score(self, qid: str, answer: str) -> dict:
        """对一题打分，返回 5 维分 + 加权总分 + 可追溯的判定明细。"""
        answer = answer or ""
        det: dict = {}
        if qid in self.key:
            spec, tier = dict(self.key[qid]), "keyed"
        else:
            final, support = self._generic_facets(qid)
            spec, tier = {"final": final, "support": support, "requires": []}, "generic"
        det["tier"] = tier

        if not answer.strip():                      # 空答卷：全维 0，不走后面的规则
            dims = {d: 0.0 for d in DIMS}
            return {"question_id": qid, **dims, "total": 0.0, "gen_len": 0,
                    "detail": {"tier": tier, "note": "空回答"}}

        views = {"full": TextView(answer),
                 "conclusion": TextView(conclusion_text(answer))}
        f = structure_features(answer)

        acc = self._accuracy(spec, views, det)
        finals = spec.get("final") or []
        acc_rate = ((sum(match_item(it, views, self.tol) for it in finals)
                     / len(finals)) if finals else 0.5)
        comp = self._completeness(spec, views, f, det)
        logic = self._logic(answer, f, acc_rate, det)
        safe = self._safety(answer, f, acc_rate, det)
        fmt = self._format(spec, f, det)

        dims = {"accuracy": acc, "completeness": comp, "logic": logic,
                "safety": safe, "format": fmt}
        total = sum(self.weights[d] * dims[d] for d in DIMS)
        return {"question_id": qid, **{k: round(v, 3) for k, v in dims.items()},
                "total": round(total, 3), "gen_len": f["len"], "detail": det}

    def score_records(self, records: list[dict]) -> dict:
        """给一整份答卷打分，返回逐题 + 按维度/按类别的汇总。"""
        per_q = [self.score(r["question_id"], r.get("answer", "")) for r in records]
        cat = {r["question_id"]: r.get("category", "?") for r in records}
        agg = {d: round(statistics.fmean([p[d] for p in per_q]), 3) for d in DIMS}
        agg["total"] = round(statistics.fmean([p["total"] for p in per_q]), 3)
        by_cat = {}
        for c in sorted({v for v in cat.values()}):
            sel = [p["total"] for p in per_q if cat[p["question_id"]] == c]
            if sel:
                by_cat[c] = round(statistics.fmean(sel), 3)
        return {"per_question": per_q, "dims": agg, "by_category": by_cat,
                "n_keyed": sum(1 for p in per_q if p["detail"].get("tier") == "keyed"),
                "n_questions": len(per_q)}


# ---------------------------------------------------------------------------
# 验证：拿 Week3 的 100 条真实人工打分当标准答案，看这套规则偏到哪去了
# ---------------------------------------------------------------------------
def spearman(a: list[float], b: list[float]) -> float:
    """Spearman 秩相关。自己实现是为了不给 step3_eval.py 引入 scipy 依赖。"""
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):                       # 处理并列名次（取平均秩）
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 3) if den else float("nan")


def load_human() -> dict:
    """读 Week3 盲测打分记录（100 行）+ 匿名映射，返回 {run_id: {qid: {dim: 分}}}。"""
    if not (HUMAN_CSV.exists() and HUMAN_MAP.exists()):
        return {}
    inv = {v: k for k, v in json.loads(HUMAN_MAP.read_text(encoding="utf-8")).items()}
    out: dict = {}
    with HUMAN_CSV.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            run = inv.get(row["匿名模型"])
            if not run:
                continue
            out.setdefault(run, {})[row["题目ID"]] = {
                "accuracy": float(row["准确性"]), "completeness": float(row["完整性"]),
                "logic": float(row["逻辑性"]), "safety": float(row["安全性"]),
                "format": float(row["格式"])}
    return out


def cmd_validate(scorer: AutoScorer) -> None:
    """把自动分和人工分并排放，报告排序一致性与逐维偏差。"""
    human = load_human()
    if not human:
        print("[fail] 找不到 Week3 人工打分记录，无法验证。")
        return
    weights = scorer.weights
    rows = []
    for run in sorted(human):
        path = WEEK3_ANSWERS / f"answers_{run}.json"
        if not path.exists():
            print(f"[skip] {run}: 缺 {path.name}")
            continue
        recs = json.loads(path.read_text(encoding="utf-8"))["records"]
        res = scorer.score_records(recs)
        h = human[run]
        hd = {d: statistics.fmean([h[q][d] for q in h]) for d in DIMS}
        ht = sum(weights[d] * hd[d] for d in DIMS)
        rows.append({"run": run, "auto": res["dims"], "human": hd,
                     "auto_total": res["dims"]["total"], "human_total": round(ht, 3),
                     "per_q": {p["question_id"]: p for p in res["per_question"]},
                     "human_q": h})

    print("\n=== 模型级：自动 vs 人工（加权总分，权重与人工卡一致）===")
    print(f"{'模型':<26}{'自动':>7}{'人工':>7}{'差':>7}   自动排名/人工排名")
    a_rank = {r['run']: i + 1 for i, r in enumerate(sorted(rows, key=lambda x: -x['auto_total']))}
    h_rank = {r['run']: i + 1 for i, r in enumerate(sorted(rows, key=lambda x: -x['human_total']))}
    for r in sorted(rows, key=lambda x: -x["human_total"]):
        print(f"{r['run']:<26}{r['auto_total']:>7.2f}{r['human_total']:>7.2f}"
              f"{r['auto_total'] - r['human_total']:>7.2f}   "
              f"{a_rank[r['run']]} / {h_rank[r['run']]}")
    rho = spearman([r["auto_total"] for r in rows], [r["human_total"] for r in rows])
    print(f"\n模型级 Spearman 秩相关 (n={len(rows)}): rho = {rho}")

    print("\n=== 维度级：全模型平均分（自动 / 人工 / 差）===")
    print(f"{'维度':<10}{'自动':>8}{'人工':>8}{'差':>8}{'题级rho':>10}")
    for d in DIMS:
        am = statistics.fmean([r["auto"][d] for r in rows])
        hm = statistics.fmean([r["human"][d] for r in rows])
        av, hv = [], []
        for r in rows:
            for q, hs in r["human_q"].items():
                if q in r["per_q"]:
                    av.append(r["per_q"][q][d])
                    hv.append(hs[d])
        print(f"{DIM_CN[d]:<10}{am:>8.2f}{hm:>8.2f}{am - hm:>8.2f}"
              f"{spearman(av, hv):>10}")

    # 题级总分相关（100 个点）
    av, hv = [], []
    for r in rows:
        for q, hs in r["human_q"].items():
            if q in r["per_q"]:
                av.append(r["per_q"][q]["total"])
                hv.append(sum(weights[d] * hs[d] for d in DIMS))
    print(f"\n题级（{len(av)} 个样本）加权总分 Spearman rho = {spearman(av, hv)}")
    print(f"题级平均绝对偏差 MAE = "
          f"{round(statistics.fmean([abs(x - y) for x, y in zip(av, hv)]), 3)}")

    # 偏差最大的 8 题，用来定位规则缺陷
    diffs = sorted(
        [(abs(sum(weights[d] * r["human_q"][q][d] for d in DIMS) - r["per_q"][q]["total"]),
          r["run"], q, r["per_q"][q]["total"],
          round(sum(weights[d] * r["human_q"][q][d] for d in DIMS), 2))
         for r in rows for q in r["human_q"] if q in r["per_q"]], reverse=True)[:8]
    print("\n=== 偏差最大的 8 条（用于定位规则缺陷）===")
    print(f"{'模型':<24}{'题':<11}{'自动':>7}{'人工':>7}{'差':>7}")
    for d_, run, q, a_, h_ in diffs:
        print(f"{run:<24}{q:<11}{a_:>7.2f}{h_:>7.2f}{d_:>7.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Week8 自动 5 维打分器")
    ap.add_argument("--answers", type=Path, help="answers_*.json 路径")
    ap.add_argument("--validate", action="store_true",
                    help="用 Week3 的 100 条人工打分验证规则（不需要 GPU）")
    ap.add_argument("--detail", action="store_true", help="逐题打印判定明细")
    args = ap.parse_args()

    scorer = AutoScorer()
    if args.validate:
        cmd_validate(scorer)
        return
    if not args.answers:
        ap.error("需要 --answers 或 --validate")
    data = json.loads(args.answers.read_text(encoding="utf-8"))
    res = scorer.score_records(data["records"])
    print(f"[{data.get('run_id', args.answers.stem)}] "
          f"精确档 {res['n_keyed']}/{res['n_questions']} 题")
    for d in DIMS:
        print(f"  {DIM_CN[d]:<6} {res['dims'][d]:.2f}")
    print(f"  加权总分 {res['dims']['total']:.2f}   分类 {res['by_category']}")
    if args.detail:
        for p in res["per_question"]:
            print(f"\n  {p['question_id']}: total={p['total']}  " +
                  "  ".join(f"{DIM_CN[d]}={p[d]}" for d in DIMS))
            print(f"    {p['detail']}")


if __name__ == "__main__":
    main()
