"""
build_react_data.py — Week6 Day31.1
构造 ReAct 格式的工具调用 SFT 训练数据（Thought/Action/Action Input/Observation）。
Build ReAct-format tool-calling SFT data.

★ 核心设计：一条轨迹 → 多条样本，且每条在 Action Input 处截断
    朴素做法是「一条完整轨迹 = 一条训练样本」，output 里含 Observation。
    这样训出来的模型会**自己编造 Observation**——因为训练数据告诉它，写完
    Action Input 之后就该接着写 Observation。而 ReAct 的 Observation 必须由框架
    真正调用工具后回填，模型一旦自己编，整个 Agent 就退化成自问自答的幻觉机器。

    正确做法：一条 n 步轨迹拆成 n 条样本。第 k 条样本的 assistant 内容 =
        前 k-1 步（含各自的 Observation，作为已知前缀）
      + 第 k 步的 Thought/Action/Action Input（**到此为止，不含 Observation**）
    于是「Action Input 之后是 EOS」这个模式被反复强化，模型学会**写完就停**。
    最后再补一条以 Final Answer 收尾的样本。

    这个拆法同时与推理时的 prefill 机制精确对齐：local_llm.py 把 scratchpad 作为
    assistant 前缀塞回去让模型续写，训练时学的正是 P(下一步 | 已有前缀) 这组条件
    概率，训练与推理完全同分布。

★ 数据配比（含负样本，防止「见工具就调」）
    | 类型                     | 条数 | 目的                                   |
    | 单步 calculator          |  ~18 | 算术必调工具                            |
    | 单步 knowledge_search    |  ~18 | 事实必查证                              |
    | 单步 code_check          |   ~8 | 代码检查走对工具                         |
    | 多步 价格+运费+预算       |  ~30 | 验收标准❷：至少 3 步                     |
    | 多步 多商品/折扣          |  ~14 | 更长链路                                |
    | 负样本 查不到            |   ~6 | 查不到要老实说，不要编、不要原样重试      |
    | 负样本 无需工具          |   ~6 | 闲聊/常识直接 Final Answer，不滥用工具    |

★ 内容即代码（沿用 Week4 make_self_built_pairs.py 的做法）
    所有轨迹由知识库真实数据程序化生成，价格/运费/总价全部由 Python 现算，
    不手写数字——保证 100% 事实正确且可复现，改知识库即自动同步。

用法 / Usage（仓库根目录）:
    .venv-agent/Scripts/python.exe Week6/code/build_react_data.py
    .venv-agent/Scripts/python.exe Week6/code/build_react_data.py --stats
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_react import REACT_PROMPT                       # noqa: E402
from local_llm import ASSIST_MARK, SYS_MARK, USER_MARK     # noqa: E402
from tools import build_all_tools                          # noqa: E402
from tools.knowledge import KnowledgeBase, _render         # noqa: E402

OUT_DIR = ROOT / "Week6" / "data"
OUT_JSON = OUT_DIR / "react_sft.json"
INFO_JSON = OUT_DIR / "dataset_info.json"
SEED = 42


# --------------------------------------------------------------------------
#  提示词还原：训练时的 system 段必须和推理时**逐字一致**
# --------------------------------------------------------------------------
def build_system_and_user(question: str) -> tuple[str, str]:
    """复用 agent_react 的模板渲染出 system / user 两段，保证训练=推理。"""
    tools = build_all_tools()
    rendered = REACT_PROMPT.format(
        sentinel_sys=SYS_MARK, sentinel_user=USER_MARK, sentinel_assist=ASSIST_MARK,
        tools="\n".join(f"{t.name}: {t.description}" for t in tools),
        tool_names=", ".join(t.name for t in tools),
        input=question, agent_scratchpad="")
    _, rest = rendered.split(SYS_MARK, 1)
    sys_part, rest = rest.split(USER_MARK, 1)
    user_part, _ = rest.split(ASSIST_MARK, 1)
    return sys_part.strip(), user_part.strip()


class Step:
    """ReAct 的一步：Thought + Action + Action Input + （回填的）Observation。"""

    def __init__(self, thought: str, action: str, action_input: str, observation: str):
        self.thought, self.action = thought, action
        self.action_input, self.observation = action_input, observation

    def head(self) -> str:
        """不含 Observation 的部分——训练样本就截断在这里。"""
        return (f"Thought: {self.thought}\n"
                f"Action: {self.action}\n"
                f"Action Input: {self.action_input}")

    def full(self) -> str:
        return f"{self.head()}\nObservation: {self.observation}"


def trajectory_to_samples(question: str, steps: list, final: str) -> list:
    """一条轨迹 → n+1 条样本，每条在 Action Input（或 Final Answer）处结束。"""
    sys_part, user_part = build_system_and_user(question)
    samples = []
    for k in range(len(steps)):
        prefix = "".join(s.full() + "\n" for s in steps[:k])
        samples.append({
            "system": sys_part,
            "instruction": user_part,
            "input": "",
            "output": prefix + steps[k].head(),
        })
    prefix = "".join(s.full() + "\n" for s in steps)
    samples.append({
        "system": sys_part,
        "instruction": user_part,
        "input": "",
        "output": prefix + f"Thought: 我现在知道最终答案了\nFinal Answer: {final}",
    })
    return samples


# --------------------------------------------------------------------------
#  轨迹生成器：全部基于知识库真实数据现算
# --------------------------------------------------------------------------
def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.2f}"


def gen_all(kb: KnowledgeBase) -> list:
    rng = random.Random(SEED)
    products = [e for e in kb.entries if e["type"] == "product"]
    docs = [e for e in kb.entries if e["type"] != "product"]
    trajs: list[tuple[str, list, str]] = []

    # ---- ① 单步 calculator ----
    pairs = [(123, 456, "*"), (899, 68, "+"), (1299, 120, "+"), (688, 25, "+"),
             (2048, 16, "/"), (37, 89, "*"), (1000, 967, "-"), (256, 4, "*"),
             (99, 101, "*")]
    for a, b, op in pairs:
        expr = f"{a} {op} {b}"
        val = _fmt(eval(expr))                        # 生成期只在本地跑，非模型输入
        q = rng.choice([f"计算 {a} {op} {b}", f"{a} {op} {b} 等于多少？",
                        f"请帮我算一下 {a} {op} {b}"])
        trajs.append((q, [Step(
            f"这是一个算术问题，我需要使用 calculator 工具精确计算，不能心算。",
            "calculator", expr, val)],
            f"{a} {op} {b} = {val}。"))

    # ---- ② 单步 knowledge_search（商品 + 政策）----
    for p in products:
        f = p["fields"]
        q = rng.choice([f"{p['title']}多少钱？", f"{p['title']}的价格是多少？",
                        f"请问{p['title']}售价多少？"])
        trajs.append((q, [Step(
            f"用户询问商品价格，我必须用 knowledge_search 查证，不能凭记忆回答。",
            "knowledge_search", p["title"], _render(p))],
            f"{p['title']}售价 {f['价格']} 元，运费 {f['运费']} 元。"))
    for d in docs:
        kw = d["keywords"][0]
        trajs.append((f"你们的{d['title']}是怎样的？", [Step(
            f"这是店铺政策问题，需要用 knowledge_search 查询准确规定。",
            "knowledge_search", kw, _render(d))],
            d["content"][:120] + "……"))

    # ---- ③ 单步 code_check ----
    snippets = [
        ("def add(a, b):\n    return a + b", "def add(a, b): return a + b 这段代码"),
        ("for i in range(10):\n    print(i)", "for 循环打印 0-9 的代码"),
        ("def f(:\n    pass", "def f(: 这段代码"),
        ("import os\nos.system('ls')", "import os 并调用 system 的代码"),
        ("x = [i**2 for i in range(5)]", "列表推导式代码"),
        ("class A:\n    def m(self):\n        return 1", "定义类 A 的代码"),
        ("while True\n    pass", "while True 缺冒号的代码"),
        ("print('hello')", "print('hello')"),
    ]
    from tools.code_executor import check_code
    for code, desc in snippets:
        # ★ Action Input 必须单行，故换行写成转义形式；**Observation 必须用这个
        #   转义后的串去算**，不能用原始多行代码算——否则训练数据里
        #   「Action Input ↔ Observation」对不上，模型学到的期望与线上真实返回不一致。
        #   Day32 实测：最初用原始代码算 Observation，导致线上 S5 死循环 6 步。
        #   （工具侧已配套支持反转义，见 code_executor.check_code。）
        escaped = code.replace("\n", "\\n")
        obs = check_code(escaped)
        trajs.append((f"请检查{desc}语法是否正确。", [Step(
            "用户要求检查 Python 代码语法，应该使用 code_check 工具做静态分析。",
            "code_check", escaped, obs)],
            obs.split("。")[0] + "。"))

    # ---- ④ 多步：价格 + 运费 + 预算（验收标准❷ 的主力）----
    for p in products:
        f = p["fields"]
        price, ship = f["价格"], f["运费"]
        total = price + ship
        for budget in (1000, 800):
            over = total > budget
            q = (f"请查询{p['title']}的价格，加上运费后计算总价，"
                 f"并告诉我是否超过预算{budget}元。")
            steps = [
                Step("我需要先查到这个商品的价格和运费，再计算总价，最后和预算比较。"
                     "第一步先用 knowledge_search 查商品信息。",
                     "knowledge_search", p["title"], _render(p)),
                Step(f"查到售价 {price} 元、运费 {ship} 元。现在用 calculator 算总价。",
                     "calculator", f"{price} + {ship}", str(total)),
                Step(f"总价是 {total} 元。现在用 calculator 判断是否超过预算 {budget}。",
                     "calculator", f"{total} > {budget}", "true" if over else "false"),
            ]
            final = (f"{p['title']}售价 {price} 元，运费 {ship} 元，总价 {total} 元，"
                     f"{'超过' if over else '没有超过'}预算 {budget} 元。")
            trajs.append((q, steps, final))

    # ---- ⑤ 多步：两件商品合计 ----
    for a, b in [(products[2], products[4]), (products[0], products[3]),
                 (products[1], products[5]), (products[3], products[4])]:
        ta = a["fields"]["价格"] + a["fields"]["运费"]
        tb = b["fields"]["价格"] + b["fields"]["运费"]
        q = f"{a['title']}和{b['title']}各买一件，一共需要多少钱（含各自运费）？"
        steps = [
            Step(f"需要分别查两件商品的价格和运费，再求和。先查{a['title']}。",
                 "knowledge_search", a["title"], _render(a)),
            Step(f"再查{b['title']}的价格和运费。",
                 "knowledge_search", b["title"], _render(b)),
            Step(f"两件商品含运费分别是 {ta} 元和 {tb} 元，用 calculator 求和。",
                 "calculator", f"{ta} + {tb}", str(ta + tb)),
        ]
        trajs.append((q, steps, f"{a['title']}含运费 {ta} 元，{b['title']}含运费 {tb} 元，"
                                f"合计 {ta + tb} 元。"))

    # ---- ⑥ 多步：会员折扣（折扣只作用于商品价，不作用于运费）----
    for p in products[:4]:
        price, ship = p["fields"]["价格"], p["fields"]["运费"]
        disc = price * 0.95
        total = disc + ship
        q = (f"我是银卡会员，买一件{p['title']}，享受折扣后加上运费一共多少钱？")
        steps = [
            Step("先查商品价格与运费，再查会员折扣规则，最后计算。",
                 "knowledge_search", p["title"], _render(p)),
            Step("需要确认银卡会员的折扣以及折扣是否作用于运费。",
                 "knowledge_search", "会员折扣", _render(
                     next(d for d in docs if "会员" in d["title"]))),
            Step(f"银卡 95 折且折扣不作用于运费，所以是 {price}*0.95+{ship}。",
                 "calculator", f"{price} * 0.95 + {ship}", _fmt(total)),
        ]
        trajs.append((q, steps,
                      f"{p['title']}售价 {price} 元，银卡 95 折后 {_fmt(disc)} 元，"
                      f"运费 {ship} 元不打折，合计 {_fmt(total)} 元。"))

    # ---- ⑦ 负样本：知识库查不到，要老实说，不要编也不要原样重试 ----
    for bogus in ["量子跃迁洗衣机", "星际穿越电视", "反重力扫地机器人"]:
        obs = _kb_miss(kb, bogus)
        trajs.append((f"{bogus}多少钱？", [Step(
            "用户询问商品价格，先用 knowledge_search 查证。",
            "knowledge_search", bogus, obs)],
            f"抱歉，知识库中没有「{bogus}」这个商品的信息，我无法提供它的价格。"
            f"目前有售的商品包括：" + "、".join(p["title"] for p in products[:3]) + " 等。"))

    # ---- ⑧ 负样本：无需工具，直接回答（防止「见工具就调」）----
    no_tool = [
        ("你好，你是谁？", "你好！我是一个可以使用工具的智能助手，"
                          "能帮你查询商品信息、做数学计算、检查 Python 代码语法。"),
        ("你能做什么？", "我可以调用三个工具：calculator 做精确计算、"
                        "knowledge_search 查商品价格与店铺政策、code_check 检查 Python 语法。"),
        ("谢谢你的帮助！", "不客气！还有其他需要帮忙的随时告诉我。"),
        ("请用一句话介绍你自己。", "我是一个能调用计算、检索和代码检查工具的智能助手。"),
    ]
    for q, a in no_tool:
        trajs.append((q, [], a))

    # 组装
    samples = []
    for q, steps, final in trajs:
        samples.extend(trajectory_to_samples(q, steps, final))
    rng.shuffle(samples)
    return samples


def _kb_miss(kb: KnowledgeBase, query: str) -> str:
    """复用工具真实的未命中输出，保证训练数据里的 Observation 与线上完全一致。"""
    from tools.knowledge import KnowledgeRetrievalTool
    return KnowledgeRetrievalTool(kb.path).run(query)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="只打印统计，不写文件")
    args = ap.parse_args()

    kb = KnowledgeBase()
    samples = gen_all(kb)

    # 统计
    n_multi = sum(1 for s in samples if s["output"].count("Action:") >= 2)
    n_final = sum(1 for s in samples if "Final Answer:" in s["output"])
    n_notool = sum(1 for s in samples
                   if "Final Answer:" in s["output"] and "Action:" not in s["output"])
    lens = [len(s["output"]) for s in samples]
    print(f"样本总数        : {len(samples)}")
    print(f"含多步前缀的样本: {n_multi}")
    print(f"以 Final Answer 收尾: {n_final}")
    print(f"无工具直接作答  : {n_notool}")
    print(f"output 长度     : 均值 {sum(lens) / len(lens):.0f}  最大 {max(lens)}")

    if args.stats:
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(samples, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    INFO_JSON.write_text(json.dumps({
        "week6_react_sft": {
            "file_name": "react_sft.json",
            "columns": {"prompt": "instruction", "query": "input",
                        "response": "output", "system": "system"},
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 -> {OUT_JSON}")
    print(f"已注册 -> {INFO_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
