"""
test_tools.py — Week6 Day28 交付
三个工具的独立单元测试（不加载大模型，纯 CPU，秒级完成）。
Standalone unit tests for the three tools — no LLM, CPU only, runs in seconds.

★ 为什么要有这一步
    Agent 出错时有两个可能来源：工具本身坏了，或模型没会用工具。若不先把工具
    钉死在「已验证」状态，Day32 的错误归因就无从谈起。本脚本把工具层的正确性
    与安全性单独验证并留档，之后 Agent 的所有异常都可归因到模型侧。

覆盖 / Coverage:
    A. Calculator 正确性   —— 四则、乘方、括号、比较、函数
    B. Calculator 安全性   —— 代码注入、属性逃逸、变量、内存炸弹（必须全部被拒）
    C. Calculator 鲁棒性   —— 全角字符、千分位、引号包裹、尾部等号
    D. Knowledge 命中      —— 商品查询、政策查询、字段意图
    E. Knowledge 未命中    —— 必须明确说「没找到」并给出可用范围
    F. Day30.2 链路预演    —— 手工串一遍三步，证明知识库数据支持该任务

用法 / Usage（仓库根目录）:
    .venv-agent/Scripts/python.exe Week6/code/test_tools.py
    .venv-agent/Scripts/python.exe Week6/code/test_tools.py --save   # 同时写交付日志
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools.calculator import CalculatorTool          # noqa: E402
from tools.knowledge import KnowledgeRetrievalTool    # noqa: E402

DELIV = ROOT / "Week6" / "deliverables"
LOG_MD = DELIV / "Day28_工具单测日志.md"

results = []       # (组, 用例, 输入, 输出, 期望, 是否通过)


def check(group: str, desc: str, got: str, predicate, expect_desc: str, inp: str):
    ok = bool(predicate(got))
    results.append((group, desc, inp, got, expect_desc, ok))
    mark = "✅" if ok else "❌"
    print(f"  {mark} {desc}\n      输入: {inp}\n      输出: {got}")
    return ok


def main(save: bool):
    calc = CalculatorTool()
    kb = KnowledgeRetrievalTool()

    print("\n=== A. Calculator 正确性 ===")
    for expr, want in [("123 * 456", "56088"), ("899 + 68", "967"),
                       ("(1299 + 120)", "1419"), ("2 ** 10", "1024"),
                       ("100 / 8", "12.5"), ("sqrt(144)", "12"),
                       ("round(3.14159, 2)", "3.14"), ("17 % 5", "2")]:
        check("A 正确性", f"{expr} = {want}", calc.run(expr),
              lambda g, w=want: g == w, f"等于 {want}", expr)

    print("\n=== B. Calculator 安全性（以下全部必须被拒绝）===")
    attacks = [
        ("__import__('os').system('dir')", "代码注入：导入 os 执行系统命令"),
        ("().__class__.__bases__", "属性逃逸：经典沙箱绕过跳板"),
        ("open('secret.txt').read()", "文件读取"),
        ("10 ** 10 ** 10", "内存炸弹：语法合法但会吃光内存"),
        ("[x for x in range(10)]", "推导式"),
        ("lambda: 1", "lambda 定义"),
        ("os.getcwd()", "属性调用"),
        ("password", "未定义变量"),
        ("1 / 0", "除零"),
    ]
    for expr, desc in attacks:
        check("B 安全性", desc, calc.run(expr),
              lambda g: g.startswith("ERROR:"), "被拒绝（返回 ERROR）", expr)

    print("\n=== C. Calculator 鲁棒性（模型格式噪声，应被磨平）===")
    for expr, want, desc in [
        ("８９９＋６８", "967", "全角数字与全角加号"),
        ("1,299 + 120", "1419", "千分位逗号"),
        ('"123 * 456"', "56088", "被引号包裹"),
        ("899 + 68 =", "967", "尾部多余等号"),
        ("（899＋68）×2", "1934", "全角括号与乘号"),
        ("2^10", "1024", "^ 当作乘方（Python 里 ^ 是异或，静默算错的经典来源）"),
    ]:
        check("C 鲁棒性", desc, calc.run(expr),
              lambda g, w=want: g == w, f"归一化后等于 {want}", expr)

    print("\n=== D. Knowledge 命中 ===")
    for q, must_have, desc in [
        ("星尘X1 价格", "899", "商品查询：星尘X1 售价"),
        ("追光S3 运费", "120", "商品查询：追光S3 运费"),
        ("退货政策", "7 天", "政策查询：退换货"),
        ("会员折扣", "95 折", "政策查询：会员等级"),
        ("云雀Pro 耳机多少钱", "459", "口语化查询"),
    ]:
        check("D 检索命中", desc, kb.run(q),
              lambda g, m=must_have: m in g, f"结果含「{must_have}」", q)

    print("\n=== E. Knowledge 未命中（必须明确说没找到）===")
    check("E 未命中", "查询不存在的商品", kb.run("量子跃迁洗衣机"),
          lambda g: "未找到" in g and "不要编造" in g,
          "明确说未找到并给出可用范围", "量子跃迁洗衣机")

    print("\n=== F. Day30.2 三步链路预演（证明知识库数据支持该任务）===")
    obs = kb.run("星尘X1 价格 运费")
    print(f"  step1 检索 -> {obs[:70]}...")
    total = calc.run("899 + 68")
    print(f"  step2 计算总价 -> {total}")
    verdict = calc.run("967 > 1000")
    print(f"  step3 比预算 -> {verdict}")
    check("F 链路预演", "星尘X1：899+68=967，未超预算 1000",
          f"{obs[:20]}|{total}|{verdict}",
          lambda g: "899" in obs and total == "967" and verdict == "false",
          "检索到 899/68，总价 967，未超预算", "三步串联")

    obs2 = kb.run("追光S3 价格 运费")
    total2 = calc.run("1299 + 120")
    verdict2 = calc.run("1419 > 1000")
    check("F 链路预演", "追光S3：1299+120=1419，超预算 1000",
          f"{total2}|{verdict2}",
          lambda g: "1299" in obs2 and total2 == "1419" and verdict2 == "true",
          "总价 1419，超预算", "三步串联")

    # ---------------- 汇总 ----------------
    passed = sum(1 for r in results if r[5])
    total_n = len(results)
    print(f"\n{'=' * 60}\n结果：{passed}/{total_n} 通过"
          f"{'  ✅ 全部通过' if passed == total_n else '  ❌ 有失败项'}\n{'=' * 60}")

    if save:
        DELIV.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Week6 Day28 交付：三工具单元测试日志", "",
            f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}　"
            f"环境：`.venv-agent`（LangChain 0.3.30）", "",
            f"**结果：{passed}/{total_n} 通过**"
            f"{'（全部通过 ✅）' if passed == total_n else '（有失败 ❌）'}", "",
            "本测试不加载大模型，纯 CPU 秒级完成。目的是把工具层的正确性与安全性",
            "钉死在「已验证」状态——之后 Agent 的任何异常都可归因到模型侧，",
            "这是 Day32 错误归因能够成立的前提。", "",
            "| # | 组别 | 用例 | 输入 | 输出 | 期望 | 结果 |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, (g, d, inp, got, exp, ok) in enumerate(results, 1):
            c = lambda s: str(s).replace("|", "\\|").replace("\n", " ")[:110]
            lines.append(f"| {i} | {g} | {c(d)} | `{c(inp)}` | `{c(got)}` "
                         f"| {c(exp)} | {'✅' if ok else '❌'} |")
        lines += ["", "## 安全性说明", "",
                  "B 组的 9 个用例是**攻击性输入**，全部必须被拒绝。工具采用**白名单 AST**",
                  "而非黑名单过滤：先 `ast.parse` 成语法树，逐节点核对类型，只有字面量、",
                  "四则/乘方/比较运算与白名单数学函数可以求值。`__import__`、属性访问、",
                  "下标、lambda、推导式在**求值之前**就被拦下，不存在被执行的路径。",
                  "`10 ** 10 ** 10` 语法合法且在白名单内，靠指数/底数限幅单独拦截。", ""]
        LOG_MD.write_text("\n".join(lines), encoding="utf-8")
        print(f"已写入 -> {LOG_MD}")

    return 0 if passed == total_n else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="同时写交付日志 md")
    sys.exit(main(ap.parse_args().save))
