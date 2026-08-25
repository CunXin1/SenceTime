"""
agent_react.py — Week6 Day29 / Day30
用 create_react_agent 构建 ReAct Agent，绑定 Day28/Day30 的三个工具。
Build a ReAct agent with create_react_agent and bind the three tools.

★ 为什么用 langchain.agents.create_react_agent（文本版）而不是 LangGraph 版
    LangChain 生态里有两个同名函数：
      - langchain.agents.create_react_agent      文本版，prompt 驱动 + 输出解析
      - langgraph.prebuilt.create_react_agent    要求模型支持 native tool calling
    本周必须用文本版，理由是 **Day31 的交付要求反向锁定了它**：31.1 要构造
    Thought/Action/Action Input/Observation 格式的训练数据——那正是文本 ReAct 的
    格式，不是 tool-calling 的 JSON 格式。若 Day29 用 tool-calling 版，Day31 微调
    出来的格式和 Agent 期望的格式对不上，整周就断成两截。
    （附带原因：本地 HF 模型走 bind_tools 支持不稳；LangChain 1.x 已删除该函数，
      故 requirements 里把 langchain 钉在 0.3.x。）

★ 三个必设的防跑飞参数
    max_iterations           上限步数。3B 模型极易在「查不到 → 换个说法再查」之间
                             反复横跳，不设上限会一直烧显存直到超时。
    handle_parsing_errors    模型输出不符合 ReAct 格式时，把解析错误作为 Observation
                             回灌，给它一次自我纠正的机会，而不是直接抛异常。
    early_stopping_method    达到上限时的收尾方式（"force" = 直接给出当前最佳答案）。
    这三个参数本身就是 Day32「死循环」失败模式的观测口——统计有多少任务是撞上限
    结束而非正常给出 Final Answer。

用法 / Usage（仓库根目录）:
    .venv-agent/Scripts/python.exe Week6/code/agent_react.py --task "计算 123 * 456"
    .venv-agent/Scripts/python.exe Week6/code/agent_react.py --suite single   # Day29 单轮
    .venv-agent/Scripts/python.exe Week6/code/agent_react.py --suite multi    # Day30 多步
    .venv-agent/Scripts/python.exe Week6/code/agent_react.py --suite all --model dpo
    #   --model base|sft|dpo 三选一做对照；--adapter 挂 Day31 的工具 SFT adapter
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain.agents import AgentExecutor, create_react_agent   # noqa: E402
from langchain_core.prompts import PromptTemplate                # noqa: E402

from local_llm import ASSIST_MARK, SYS_MARK, USER_MARK, LocalQwenLLM  # noqa: E402
from tools import build_all_tools                                # noqa: E402

DELIV = ROOT / "Week6" / "deliverables"

# 三个候选 policy，Day29 用 dpo，Day32 做三方对照。
MODELS = {
    "base": "models/Qwen2.5-3B-Instruct",
    "sft": "models/Qwen2.5-3B-week3-best-merged",
    "dpo": "models/Qwen2.5-3B-week4-dpo-merged",
}

# --------------------------------------------------------------------------
#  ReAct 提示词
#  哨兵 <<<SYSTEM>>>/<<<USER>>>/<<<ASSISTANT>>> 由 local_llm.split_prompt 还原成
#  chat 三段式；{agent_scratchpad} 落在 ASSISTANT 段，作为「续写前缀」。
#  详见 local_llm.py 抬头对 prefill 的说明。
# --------------------------------------------------------------------------
REACT_PROMPT = """{sentinel_sys}
你是一个能使用工具的智能助手。你必须严格按照下面的格式回答问题。

可用工具：
{tools}

你必须使用以下格式，且每次只输出到 Action Input 为止，然后停下来等待 Observation：

Question: 需要回答的问题
Thought: 你的思考，说明下一步要做什么、为什么
Action: 要使用的工具名，必须是 [{tool_names}] 中的一个，只写工具名，不要写别的
Action Input: 传给该工具的输入
Observation: 工具返回的结果（这一行由系统填写，你绝对不要自己编写）
...（Thought/Action/Action Input/Observation 可以重复多次）
Thought: 我现在知道最终答案了
Final Answer: 对原问题的完整中文回答

重要规则：
1. 任何算术（加减乘除、比较大小）都必须调用 calculator，绝对不要心算。
2. 任何商品价格、运费、店铺政策都必须调用 knowledge_search 查证，绝对不要凭记忆编造。
3. 写完 Action Input 后必须立刻停止，等待系统返回 Observation，不要自己编造 Observation。
4. 拿到足够信息后，必须以 "Final Answer:" 开头给出最终答案。
5. 不要重复调用同一个工具做同样的查询；如果一个查询没查到，换个关键词，不要原样重试。

{sentinel_user}
Question: {input}

{sentinel_assist}
{agent_scratchpad}"""


def build_prompt() -> PromptTemplate:
    """把哨兵注入模板。用 partial 而不是直接写进字符串，避免与 {} 占位冲突。"""
    return PromptTemplate.from_template(REACT_PROMPT).partial(
        sentinel_sys=SYS_MARK, sentinel_user=USER_MARK, sentinel_assist=ASSIST_MARK)


def build_agent(model_key: str = "dpo", adapter: str | None = None,
                max_iterations: int = 6, verbose: bool = True,
                max_new_tokens: int = 512) -> AgentExecutor:
    model_path = MODELS.get(model_key, model_key)
    print(f"[加载] policy = {model_path}" + (f"  + adapter {adapter}" if adapter else ""))
    llm = LocalQwenLLM.build(model_path, adapter=adapter, max_new_tokens=max_new_tokens)
    tools = build_all_tools()
    print(f"[工具] {[t.name for t in tools]}")

    agent = create_react_agent(llm, tools, build_prompt())
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,      # 防死循环（Day32 观测口）
        handle_parsing_errors=True,         # 格式错时回灌错误，给一次自我纠正机会
        early_stopping_method="force",
        return_intermediate_steps=True,     # 留档给 Day32 做失败归因
    )


# --------------------------------------------------------------------------
#  测试题集
# --------------------------------------------------------------------------
SINGLE_TASKS = [   # Day29.3 单轮工具调用
    {"id": "S1", "task": "计算 123 * 456", "expect": "56088", "tool": "calculator"},
    {"id": "S2", "task": "星尘X1智能音箱多少钱？", "expect": "899", "tool": "knowledge_search"},
    {"id": "S3", "task": "计算 2 的 10 次方", "expect": "1024", "tool": "calculator"},
    {"id": "S4", "task": "你们的退货政策是几天？", "expect": "7", "tool": "knowledge_search"},
    {"id": "S5", "task": "检查这段Python代码语法是否正确：def add(a, b): return a + b",
     "expect": "语法正确", "tool": "code_check"},
]

MULTI_TASKS = [    # Day30.2 多步复杂任务（每题至少 3 步）
    {"id": "M1",
     "task": "请查询星尘X1智能音箱的价格，加上运费后计算总价，并告诉我是否超过预算1000元。",
     "expect": "967", "min_steps": 3},
    {"id": "M2",
     "task": "请查询追光S3显示器的价格，加上运费后计算总价，并告诉我是否超过预算1000元。",
     "expect": "1419", "min_steps": 3},
    {"id": "M3",
     "task": "磐石M2机械键盘和微澜A1加湿器各买一件，一共需要多少钱（含各自运费）？",
     "expect": "927", "min_steps": 3},
    {"id": "M4",
     "task": "云雀Pro无线耳机的总价（含运费）是多少？如果我是银卡会员享95折，"
             "折后商品价加运费一共多少钱？",
     "expect": "436", "min_steps": 3},
]


def run_task(executor: AgentExecutor, task: dict) -> dict:
    """跑一道题，返回含中间步骤的完整记录。"""
    t0 = time.perf_counter()
    record = {"id": task["id"], "task": task["task"], "expect": task.get("expect")}
    try:
        out = executor.invoke({"input": task["task"]})
        answer = out.get("output", "")
        steps = out.get("intermediate_steps", [])
        record["answer"] = answer
        record["steps"] = [
            {"tool": a.tool, "input": str(a.tool_input)[:200], "observation": str(o)[:300]}
            for a, o in steps
        ]
        record["n_steps"] = len(steps)
        record["tools_used"] = [a.tool for a, _ in steps]
        # 判定：期望值出现在最终答案里即算命中（宽松判定，人工复核见 Day32）
        record["hit"] = bool(task.get("expect")) and task["expect"] in answer.replace(",", "")
        # 撞上限而非正常收尾 —— Day32 的「死循环」计数口径
        record["hit_limit"] = "Agent stopped due to iteration limit" in answer or \
                              "stopped due to max iterations" in answer.lower()
    except Exception as exc:                      # noqa: BLE001
        record.update(answer=f"EXCEPTION: {type(exc).__name__}: {exc}",
                      steps=[], n_steps=0, tools_used=[], hit=False, hit_limit=False)
    record["seconds"] = round(time.perf_counter() - t0, 1)
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="直接跑一道自定义任务")
    ap.add_argument("--suite", choices=["single", "multi", "all"], help="跑内置题集")
    ap.add_argument("--model", default="dpo", help="base / sft / dpo，或直接给路径")
    ap.add_argument("--adapter", default=None, help="可选：Day31 工具 SFT adapter 路径")
    ap.add_argument("--tag", default=None, help="结果文件后缀，用于区分对照组")
    ap.add_argument("--max-iterations", type=int, default=6)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    executor = build_agent(args.model, args.adapter, args.max_iterations,
                           verbose=not args.quiet)

    tasks = []
    if args.task:
        tasks = [{"id": "custom", "task": args.task}]
    elif args.suite in ("single", "all"):
        tasks += SINGLE_TASKS
    if args.suite in ("multi", "all"):
        tasks += MULTI_TASKS
    if not tasks:
        ap.error("必须给 --task 或 --suite")

    records = []
    for i, t in enumerate(tasks, 1):
        print(f"\n{'#' * 70}\n# [{i}/{len(tasks)}] {t['id']}: {t['task']}\n{'#' * 70}")
        rec = run_task(executor, t)
        records.append(rec)
        print(f"\n>> 答案: {rec['answer'][:300]}")
        print(f">> 步数: {rec['n_steps']}  工具: {rec['tools_used']}  "
              f"命中: {rec['hit']}  耗时: {rec['seconds']}s")

    # 汇总
    n = len(records)
    hits = sum(1 for r in records if r.get("hit"))
    loops = sum(1 for r in records if r.get("hit_limit"))
    print(f"\n{'=' * 70}\n题数 {n}　命中 {hits}/{n}　撞步数上限 {loops}　"
          f"平均步数 {sum(r['n_steps'] for r in records) / max(n, 1):.1f}\n{'=' * 70}")
    print(f"LLM 调用统计: {LocalQwenLLM.call_stats}")

    tag = args.tag or f"{args.model}{'_lora' if args.adapter else ''}"
    DELIV.mkdir(parents=True, exist_ok=True)
    out_path = DELIV / f"agent_run_{args.suite or 'custom'}_{tag}.json"
    out_path.write_text(json.dumps({
        "meta": {"model": args.model, "adapter": args.adapter,
                 "max_iterations": args.max_iterations,
                 "time": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                 "llm_stats": LocalQwenLLM.call_stats},
        "summary": {"n": n, "hits": hits, "hit_limit": loops},
        "records": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
