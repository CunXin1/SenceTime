"""
analyze_errors.py — Week6 Day32.1
统计 Agent 的失败模式，生成《Agent错误模式分析报告》。
Classify Agent failure modes and emit the error-analysis report.

★ 归因的前提：工具层已被证明是对的
    Day28 的 31/31 单测把工具的正确性与安全性钉死了，所以本脚本观察到的任何异常
    都可以归因到**模型侧**。没有那一步，「选错工具」和「工具本身召回错」是分不开的。

★ 五类失败模式（任务书 32.1 要求统计前三类，这里多加两类）
    1. wrong_tool     选错工具：该查知识库却去算数，或该算数却去查知识库
    2. bad_args       参数提取错误：工具返回 ERROR / 未找到，说明 Action Input 不合法
    3. dead_loop      死循环：撞 max_iterations，或对同一工具重复发出完全相同的输入
    4. format_error   格式错误：输出不符合 ReAct 格式，触发 handle_parsing_errors
    5. hallucination  幻觉：没调该调的工具就直接给答案，或最终数字与真值不符

    区分 1 与 5 很重要：选错工具是「调了但调错」，幻觉是「压根没调」。
    前者靠改工具描述治，后者靠改 System Prompt 或做 SFT 治——手段完全不同。

用法 / Usage（仓库根目录）:
    .venv-agent/Scripts/python.exe Week6/code/analyze_errors.py
    .venv-agent/Scripts/python.exe Week6/code/analyze_errors.py --runs dpo base sft
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DELIV = ROOT / "Week6" / "deliverables"
REPORT = DELIV / "Agent错误模式分析报告.md"

# 每道题「应该」用到的工具，用于判定选错工具/幻觉
EXPECTED = {
    "S1": {"calculator"}, "S2": {"knowledge_search"}, "S3": {"calculator"},
    "S4": {"knowledge_search"}, "S5": {"code_check"},
    "M1": {"knowledge_search", "calculator"}, "M2": {"knowledge_search", "calculator"},
    "M3": {"knowledge_search", "calculator"}, "M4": {"knowledge_search", "calculator"},
}
MODE_CN = {
    "wrong_tool": "选错工具", "bad_args": "参数提取错误", "dead_loop": "死循环",
    "format_error": "格式错误", "hallucination": "幻觉/漏调工具",
    "redundant_call": "重复调用", "lazy_calc": "该调计算器却心算",
}

# ★ 严格判定表（Day30 实测后补）
#   agent_react.py 里的 hit 判定是「期望值出现在答案里」，太松：M4 的答案首句
#   「总价（含运费）是 436.05 元」是**事实错误**（含运费总价应为 459，436.05 是折后价），
#   但因为答案里含 "436" 就被判成命中。这里事后统一重判，must 里的每一项都必须出现。
#   —— 判定口径本身就是 Day32 的一条方法论发现：宽松判定会掩盖真实错误率。
STRICT = {
    "S1": ["56088"], "S2": ["899"], "S3": ["1024"], "S4": ["7"], "S5": ["语法正确"],
    "M1": ["967"], "M2": ["1419"], "M3": ["927"],
    "M4": ["459", "436"],          # 含运费总价 459 与折后价 436.05 必须都说对
}
# 这些题的最后一步「比大小」应当交给 calculator，心算即记 lazy_calc
NEED_COMPARE = {"M1", "M2"}


CONCLUSION = """
## 四、深度分析：`dpo_lora_v1` 的死循环与其根因

Day31 的工具调用 SFT **第一版反而让总命中率从 8/9 掉到 7/9**，并新增一例死循环。
把这个负面结果查到底，是本周最有价值的一段工作。

### 4.1 现象

S5（检查 `def add(a, b): return a + b` 的语法）在微调前 1 步答对，微调后
**连续 6 次发出完全相同的 `code_check` 调用**，直到撞 `max_iterations` 被强制中止：

```
lora[1..6] code_check <- def add(a, b):\\n    return a + b     ← 六次完全一致
Observation: 语法错误：第 1 行第 16 列：unexpected character after
             line continuation character。
```

### 4.2 根因：训练数据里 Action Input 与 Observation 不自洽

ReAct 的 `Action Input` 是**单行**的——模型无法在其中输出真正的换行，只能写转义的
`\\n`。构造训练数据时这一点处理对了，但 **Observation 却是用「真实换行的原始代码」
算出来的**：

| | 训练数据（v1） | 线上真实 |
|---|---|---|
| Action Input | `def add(a, b):\\n    return a + b`（字面 `\\n`） | 同左 |
| Observation | `语法正确（共 2 行）` ← 用**真换行**代码算的 | `语法错误：…line continuation…` |

于是模型学到的因果是「发出这个 Action Input → 会得到『语法正确』」。线上却拿到语法
错误，与习得的预期冲突；模型不认为是自己写错了，于是**原样重试**——而重试必然得到
同样的错误，形成闭环。

> **这不是模型的问题，是数据构造的问题。** 值得强调的是：v1 的训练 loss 低到 0.0132，
> 从损失曲线上完全看不出任何异常——**训练指标健康与线上行为正确是两回事**，
> 只有端到端跑 Agent 才能暴露这类分布不一致。

### 4.3 修复

两个方向：让模型别写转义（做不到，ReAct 格式所限），或**让工具接受转义**。选后者：

1. `tools/code_executor.py`：当输入含字面 `\\n` 且不含真换行时，反转义成真换行。
   这与 CalculatorTool 磨平全角字符是**同一类输入归一化**——把模型受格式所限而
   必然产生的写法，还原成工具真正需要的形式。
2. `build_react_data.py`：Observation 改为**用转义后的那个串**去调真实工具计算，
   保证训练数据里「Action Input ↔ Observation」严格自洽。

### 4.4 一条可复用的原则

> **训练数据里的 Observation，必须由「Action Input 里那个一模一样的字符串」
> 喂给真实工具产生，绝不能用另一份等价但不同形的输入去算。**

任何"等价改写"（转义、去空格、格式化）都会在训练与推理之间制造分布裂缝，而这种裂缝
在 loss 上不可见，只会在端到端行为上炸开。

### 4.5 修复后：9/9，全部失败模式清零

用修正后的数据重训（超参一字未改），端到端重测：

| 组 | 严格命中 | 多步题 | 平均步数 | 失败模式 |
|---|---|---|---|---|
| `dpo`（微调前） | 8/9 | 3/4 | 2.0 | 选错工具1、心算1、重复调用1、幻觉1 |
| `dpo_lora_v1`（数据有 bug） | **7/9** ↓ | 3/4 | 2.4 | **死循环1**、重复调用1、幻觉1 |
| `dpo_lora`（修复后） | **9/9** ✅ | **4/4** ✅ | 1.8 | **无** |

S5 从 6 步死循环回到 **1 步答对**，工具正确返回「语法正确（共 2 行）」——
证明修复在端到端链路上真正生效，而非只是让指标好看。

> **最值得记住的一点**：v1 与 v2 的训练指标几乎完全相同
> （train loss 0.0132 vs 0.0131，eval loss 0.1964 vs 0.1935）。
> 一个会导致线上死循环的数据 bug，在损失曲线上**完全不可见**。
> Agent 的质量只能靠端到端跑任务来衡量。

## 五、各失败模式的治法与实测效果

| 失败模式 | 该用什么治 | 本周实测 |
|---|---|---|
| 该调计算器却心算（`lazy_calc`） | **SFT** | ✅ M2 从 2 步（心算比大小）→ 3 步（调 calculator 比） |
| 重复调用（`redundant_call`） | **SFT** | ✅ M3 从 5 步（4 次检索）→ 3 步（2 次检索） |
| 死循环（`dead_loop`） | **训练数据自洽** + `max_iterations` 兜底 | ✅ 修复数据后清零；上限保证了即使复发也能收敛而非挂死 |
| 幻觉/表述错误（`hallucination`） | **SFT** | ✅ M4 从"把折后价当含运费总价"→ 两个事实分别说对 |
| 选错工具（`wrong_tool`） | 改工具名 / 改工具描述 | ✅ 45 次运行中 0 次把 `code_check` 当解释器用 |

### 关于工具命名的验证

第三个工具取名 **`code_check` 而非 `code_executor`**（任务书里叫 CodeExecutor）。
五组共 45 次任务运行中，**没有任何一次**模型试图用它求代码的运行结果——
名字的字面语义确实主导了模型的工具选择。这条改动的收益无法从描述文本上看出来，
但它消除了整整一类潜在失败。

### M4：一个被修正的预判

「云雀Pro 的总价（含运费）是多少？如果我是银卡会员享 95 折，折后一共多少钱？」
这道题含**两个独立问句**，微调前所有组都只答对一半：

- `dpo`：答"总价（含运费）是 436.05 元"——把折后价当成了含运费总价（事实错误）
- `dpo_lora_v1`：只查不算，且折扣算错（说 444.05，实为 436.05）

分析初期我判断这是**结构性局限**：ReAct 的 scratchpad 没有"待办清单"机制，
答完第一问就会认为任务完成，需要在编排层做任务分解才能解决。

**实测推翻了这个判断。** 修复数据后的 `dpo_lora` 给出：
「售价 459 元，含运费；银卡会员享 95 折后折后价格为 436.05 元」——两个事实分别说对。
说明只要训练数据里有足够多"一次答完多个子问题"的样本（本周的多步轨迹本身提供了
这种模式），3B 模型是可以在单条 scratchpad 内维持多问句的。
**先下"结构性局限"的结论下早了；能靠数据解决的问题，不要急着归因到架构。**

## 六、方法论发现：判定口径会系统性低估错误率

`agent_react.py` 运行时的宽松判定（"期望值出现在答案里"）给出 dpo **9/9**，
而严格判定（"每个必需事实都必须说对"）只有 **8/9**——差距正是 M4 那条
"含正确数字但表述错误"的答案：模型说"总价（含运费）是 436.05 元"，
里面确实有 `436`，但这句话本身是错的（含运费总价应为 459）。

若只看宽松口径，会得出"微调前就已经 9/9、没什么可优化"的结论，从而**既错过 M4 的
真实错误，也测不出微调的收益**（修复后的 dpo_lora 在两种口径下都是 9/9，
但只有严格口径能显示出它相对 dpo 的进步）。

**Agent 评测里，"答案里出现了正确数字"离"答对了"还有很远的距离。**
自动判定必须要求每个必需事实都出现，且最好再人工抽检表述是否成立。
"""


def strict_hit(rec: dict) -> bool:
    """严格判定：STRICT 里列出的每一项都必须出现在最终答案里。"""
    must = STRICT.get(rec["id"])
    if not must:
        return bool(rec.get("hit"))
    ans = str(rec.get("answer", "")).replace(",", "")
    return all(m in ans for m in must)


def classify(rec: dict) -> list:
    """对一条运行记录判定失败模式，可同时命中多类。"""
    modes = []
    used = set(rec.get("tools_used") or [])
    want = EXPECTED.get(rec["id"], set())
    steps = rec.get("steps") or []

    # 3) 死循环：撞步数上限
    seen = Counter((s["tool"], s["input"]) for s in steps)
    if rec.get("hit_limit"):
        modes.append("dead_loop")
    # 6) 重复调用：同一 (工具, 输入) 出现 ≥2 次，或同一工具查了同一对象两遍。
    #    与死循环分开计：重复调用只是浪费步数，任务仍可完成；死循环是任务失败。
    if seen and max(seen.values()) >= 2:
        modes.append("redundant_call")
    elif len(steps) >= 4 and len([s for s in steps if s["tool"] == "knowledge_search"]) > 2:
        modes.append("redundant_call")

    # 7) 该调计算器却心算：M1/M2 要求比大小，但 calculator 只被调了一次（仅算总价）
    if rec["id"] in NEED_COMPARE:
        n_calc = len([s for s in steps if s["tool"] == "calculator"])
        if n_calc < 2:
            modes.append("lazy_calc")

    # 2) 参数提取错误：工具明确返回 ERROR 或未找到
    if any(str(s["observation"]).startswith("ERROR:") or "未找到" in str(s["observation"])
           for s in steps):
        modes.append("bad_args")

    # 4) 格式错误：解析失败会被 handle_parsing_errors 回灌成特定文案
    blob = json.dumps(rec, ensure_ascii=False)
    if re.search(r"Invalid Format|could not parse|Missing 'Action|解析", blob, re.I):
        modes.append("format_error")

    # 1) 选错工具：用了不在期望集合里的工具
    if want and (used - want):
        modes.append("wrong_tool")

    # 5) 幻觉/漏调：该调的工具一个没调，或严格判定不通过
    if want and not (used & want):
        modes.append("hallucination")
    elif not strict_hit(rec) and not rec.get("hit_limit"):
        modes.append("hallucination")

    return list(dict.fromkeys(modes))


def load_runs(tags: list) -> dict:
    runs = {}
    for tag in tags:
        recs, meta = [], {}
        for suite in ("single", "multi", "all"):
            p = DELIV / f"agent_run_{suite}_{tag}.json"
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                recs.extend(d["records"])
                meta = d.get("meta", meta)
        if recs:
            # 同一题可能在 single 和 all 里各出现一次，按 id 去重保留最后一次
            dedup = {r["id"]: r for r in recs}
            runs[tag] = {"records": list(dedup.values()), "meta": meta}
        else:
            print(f"[跳过] 没有找到 {tag} 的运行结果")
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["base", "sft", "dpo", "dpo_lora"],
                    help="要对比的 tag（对应 agent_run_*_<tag>.json）")
    args = ap.parse_args()

    runs = load_runs(args.runs)
    if not runs:
        print("没有任何运行结果，先跑 agent_react.py --suite all")
        return 1

    # 逐组统计
    stats = {}
    for tag, d in runs.items():
        recs = d["records"]
        modes = Counter()
        per_task = {}
        for r in recs:
            m = classify(r)
            per_task[r["id"]] = m
            modes.update(m)
        multi = [r for r in recs if r["id"].startswith("M")]
        stats[tag] = {
            "n": len(recs),
            "hits": sum(1 for r in recs if strict_hit(r)),
            "loose_hits": sum(1 for r in recs if r.get("hit")),
            "modes": modes,
            "per_task": per_task,
            "avg_steps": sum(r["n_steps"] for r in recs) / max(len(recs), 1),
            "multi_ok": sum(1 for r in multi if strict_hit(r)),
            "multi_n": len(multi),
            "multi_3step": sum(1 for r in multi if r["n_steps"] >= 3),
            "avg_sec": sum(r.get("seconds", 0) for r in recs) / max(len(recs), 1),
        }

    # ---------------- 报告 ----------------
    L = ["# Week6 Day32 交付：Agent 错误模式分析报告", "",
         f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}　环境：`.venv-agent`（LangChain 0.3.30）", "",
         "## 〇、归因的前提", "",
         "Day28 的 31/31 工具单测（含 9 项安全攻击全部被拒）已经把**工具层**的正确性钉死。",
         "因此本报告观察到的所有异常都可归因到**模型侧**——若没有那一步，",
         "「模型选错工具」与「工具自身召回错误」是无法区分的。", "",
         "## 一、总体对比", "",
         "> **命中率按严格口径统计**：`agent_react.py` 运行时的宽松判定只看「期望值是否",
         "> 出现在答案里」，会放过 M4 这类「答案里含正确数字、但表述是事实错误」的情况。",
         "> 下表的「宽松」列保留原口径以显示两者差距——**判定口径本身就是一条方法论发现：",
         "> 宽松判定会系统性低估错误率**。", "",
         "| 组 | 题数 | 严格命中 | 严格命中率 | 宽松命中 | 多步题命中 | 达成≥3步 | 平均步数 | 平均耗时 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for tag, s in stats.items():
        L.append(f"| `{tag}` | {s['n']} | {s['hits']} | {s['hits'] / max(s['n'], 1):.0%} "
                 f"| {s['loose_hits']} "
                 f"| {s['multi_ok']}/{s['multi_n']} | {s['multi_3step']}/{s['multi_n']} "
                 f"| {s['avg_steps']:.1f} | {s['avg_sec']:.0f}s |")

    L += ["", "## 二、失败模式分布", "",
          "| 失败模式 | " + " | ".join(f"`{t}`" for t in stats) + " | 说明 |",
          "|---|" + "---|" * len(stats) + "---|"]
    why = {
        "wrong_tool": "调了工具但调错对象——靠**改工具描述**治",
        "bad_args": "Action Input 不合法导致工具报错——靠**归一化 + 可自愈报错**治",
        "dead_loop": "撞步数上限或重复同一查询——靠**未命中提示 + max_iterations** 治",
        "format_error": "输出不符合 ReAct 格式——靠 **SFT** 治，提示词收效有限",
        "hallucination": "压根没调该调的工具、或最终表述与事实不符——靠 **SFT** 治",
        "redundant_call": "同一查询重复发起，浪费步数但任务仍可完成——靠**提示词**治",
        "lazy_calc": "比大小等简单算术自己心算，违反「必须调 calculator」——靠 **SFT** 治",
    }
    for mode in MODE_CN:
        cells = " | ".join(str(stats[t]["modes"].get(mode, 0)) for t in stats)
        L.append(f"| {MODE_CN[mode]} | {cells} | {why[mode]} |")

    L += ["", "## 三、逐题失败明细", "",
          "| 题号 | " + " | ".join(f"`{t}`" for t in stats) + " |",
          "|---|" + "---|" * len(stats)]
    all_ids = sorted({i for s in stats.values() for i in s["per_task"]},
                     key=lambda x: (x[0], x))
    for tid in all_ids:
        cells = []
        for t in stats:
            m = stats[t]["per_task"].get(tid)
            cells.append("✅" if m == [] else ("／" if m is None
                         else "、".join(MODE_CN[x] for x in m)))
        L.append(f"| {tid} | " + " | ".join(cells) + " |")

    L += CONCLUSION.splitlines()

    DELIV.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")

    for tag, s in stats.items():
        print(f"[{tag}] 命中 {s['hits']}/{s['n']}　多步 {s['multi_ok']}/{s['multi_n']}　"
              f"失败模式 {dict(s['modes'])}")
    print(f"已写入 -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
