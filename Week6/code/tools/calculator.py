"""
calculator.py — Week6 Day28.2
CalculatorTool：安全沙箱执行数学表达式。
CalculatorTool: evaluate math expressions inside an AST whitelist sandbox.

★ 为什么不用 eval() / numexpr
    `eval("__import__('os').system('rm -rf /')")` 是一行就能拿下宿主机的经典洞。
    本工具的输入直接来自 LLM 的 Action Input——那是**不可信输入**（模型可能被
    prompt injection 诱导，也可能自己胡乱拼接）。因此这里不做「黑名单过滤」
    （黑名单永远漏，如 `().__class__.__bases__` 这类绕过），而是走**白名单 AST**：
    先 ast.parse 成语法树，逐节点检查类型，只有在白名单里的节点才允许求值，
    其余一律拒绝。这样即使模型写出 import / 属性访问 / 下标 / lambda，都在
    求值前就被挡掉，根本不会执行到。

★ 三层防护 / three layers
    1) 节点白名单：只放行字面量、二元/一元运算、括号、以及白名单函数调用。
       禁止 Name（变量）、Attribute（属性）、Subscript（下标）、Call 到非白名单、
       Lambda、推导式、赋值表达式（海象）等。
    2) 幂运算限幅：`10**10**10` 语法完全合法、白名单也放行，但会瞬间吃光内存
       （CPython 大整数没有上限）。故对 ** 的底数/指数做量级限制。
    3) 结果与中间量限幅：整数位数超限直接报错，避免返回一个几十万位的数把
       上下文撑爆（对 Agent 来说，撑爆 context 等价于任务失败）。

★ 输入归一化 / input normalization
    中文模型很爱输出全角字符（２３４、＋－×÷、（）)与千分位逗号，还常在表达式
    尾巴上带个「=」或包一层引号。这些不是模型「算错」，是格式噪声。工具层做保守
    归一化（全角→半角、×÷→*/、去千分位、去尾部=与引号），并把归一化前后的差异
    记录在 last_normalization 里——Day32 做错误模式分析时要靠它区分
    「参数提取错误」和「单纯格式噪声」，两者的优化手段完全不同。

用法 / Usage:
    from tools.calculator import CalculatorTool
    tool = CalculatorTool()
    tool.run("123 * 456")            # -> '56088'
    tool.run("899 + 68")             # -> '967'
    tool.run("__import__('os')")     # -> 'ERROR: ...不支持的语法...'
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, ClassVar, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# 白名单：运算符
# --------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
# 比较运算：Day30.2 的「是否超过预算 1000」需要它，否则模型得自己判断大小，
# 而 3B 模型对「967 是否 > 1000」的自答错误率显著高于让它调工具算。
_CMP_OPS = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}

# 白名单：可调用函数。只放纯数学、无副作用、无文件/网络访问的。
_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "int": int, "float": float,
    "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil,
    "log": math.log, "log2": math.log2, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "degrees": math.degrees, "radians": math.radians, "fabs": math.fabs,
}
# 白名单：常量。只有这三个 Name 允许出现，其余变量名一律拒绝。
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

# 限幅参数（见上文「三层防护」）
_MAX_POW_EXP = 256          # 指数绝对值上限
_MAX_POW_BASE = 10 ** 15    # 底数绝对值上限
_MAX_INT_DIGITS = 100       # 整数结果位数上限

# 输入归一化表：全角 → 半角，以及中文常用运算符
_NORMALIZE = str.maketrans({
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "＋": "+", "－": "-", "−": "-", "—": "-", "×": "*", "✕": "*", "＊": "*",
    "÷": "/", "／": "/", "％": "%", "＾": "^",
    "（": "(", "）": ")", "．": ".", "＝": "=", "，": ",", "　": " ",
})


class UnsafeExpression(ValueError):
    """表达式含白名单之外的语法/调用，拒绝求值。"""


def normalize_expression(raw: str) -> str:
    """把模型输出的格式噪声磨平，不改变数学语义。"""
    s = str(raw).strip()
    # 模型常把 Action Input 包在引号或反引号里
    for q in ('"', "'", "`"):
        if len(s) >= 2 and s.startswith(q) and s.endswith(q):
            s = s[1:-1].strip()
    s = s.translate(_NORMALIZE)
    # 「计算 123*456 = 」这类尾巴
    s = s.rstrip("=").strip()
    # 千分位逗号：仅当形如 1,234 / 12,345,678 时去掉，避免误伤 min(1,2)
    import re
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    # ^ 在数学书写里是乘方，但在 Python 里是异或——这是静默算错的经典来源
    s = s.replace("^", "**")
    return s


def _guard_pow(base: Any, exp: Any) -> None:
    """拦 10**10**10 这类「语法合法但会打爆内存」的表达式。"""
    if isinstance(exp, (int, float)) and abs(exp) > _MAX_POW_EXP:
        raise UnsafeExpression(
            f"指数 {exp} 超出上限 {_MAX_POW_EXP}，已拒绝（防止内存耗尽）")
    if isinstance(base, (int, float)) and abs(base) > _MAX_POW_BASE:
        raise UnsafeExpression(
            f"底数 {base} 超出上限 {_MAX_POW_BASE}，已拒绝（防止内存耗尽）")


def _guard_result(value: Any) -> Any:
    """结果限幅：几十万位的整数会把 Agent 的 context 撑爆。"""
    if isinstance(value, int) and not isinstance(value, bool):
        if len(str(abs(value))) > _MAX_INT_DIGITS:
            raise UnsafeExpression(
                f"结果整数位数超过 {_MAX_INT_DIGITS} 位，已拒绝（防止撑爆上下文）")
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        raise UnsafeExpression(f"结果为 {value}，非有限数值")
    return value


def _eval_node(node: ast.AST) -> Any:
    """递归求值。白名单之外的节点类型一律抛 UnsafeExpression。"""
    # --- 字面量 ---
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise UnsafeExpression(f"不支持的字面量类型：{type(node.value).__name__}")

    # --- 括号内的表达式 / 二元运算 ---
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise UnsafeExpression(f"不支持的运算符：{op_type.__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if op_type is ast.Pow:
            _guard_pow(left, right)
        try:
            return _guard_result(_BIN_OPS[op_type](left, right))
        except ZeroDivisionError:
            raise UnsafeExpression("除数为 0")

    # --- 一元运算 ---
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise UnsafeExpression(f"不支持的一元运算符：{op_type.__name__}")
        return _guard_result(_UNARY_OPS[op_type](_eval_node(node.operand)))

    # --- 比较（含链式 1 < x < 10）---
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in _CMP_OPS:
                raise UnsafeExpression(f"不支持的比较运算符：{op_type.__name__}")
            right = _eval_node(comparator)
            if not _CMP_OPS[op_type](left, right):
                return False
            left = right
        return True

    # --- 白名单函数调用 ---
    if isinstance(node, ast.Call):
        # 只允许 f(...) 形式的直接调用；obj.method(...) 一律拒绝
        # （Attribute 是沙箱逃逸最常见的跳板）
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpression("不支持属性调用（如 obj.method()）")
        fname = node.func.id
        if fname not in _FUNCS:
            raise UnsafeExpression(
                f"不支持的函数：{fname}（可用：{', '.join(sorted(_FUNCS))}）")
        if node.keywords:
            raise UnsafeExpression("不支持关键字参数")
        args = [_eval_node(a) for a in node.args]
        if fname == "pow" and len(args) >= 2:
            _guard_pow(args[0], args[1])
        try:
            return _guard_result(_FUNCS[fname](*args))
        except (ValueError, OverflowError, TypeError) as exc:
            raise UnsafeExpression(f"{fname} 调用失败：{exc}")

    # --- 白名单常量 ---
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise UnsafeExpression(
            f"不支持的变量名：{node.id}（本工具只算纯数字表达式，"
            f"可用常量：{', '.join(sorted(_CONSTS))}）")

    # --- 兜底：Attribute/Subscript/Lambda/推导式/海象 等全部落到这里 ---
    raise UnsafeExpression(f"不支持的语法节点：{type(node).__name__}")


def safe_eval(expression: str) -> Any:
    """对外的纯函数入口：归一化 → 解析 → 白名单求值。"""
    expr = normalize_expression(expression)
    if not expr:
        raise UnsafeExpression("表达式为空")
    if len(expr) > 500:
        raise UnsafeExpression("表达式过长（>500 字符），已拒绝")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"语法错误：{exc.msg}")
    return _eval_node(tree.body)


def format_number(value: Any) -> str:
    """整数就输出整数，避免 967.0 这种让模型误以为「不是整数」的写法。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


class CalculatorInput(BaseModel):
    expression: str = Field(
        description=(
            "纯数字数学表达式，不要带中文、不要带单位、不要带等号。"
            "例如 123 * 456、899 + 68、(1299 + 120) > 1000"
        )
    )


class CalculatorTool(BaseTool):
    """LangChain 工具：安全计算数学表达式。"""

    name: str = "calculator"
    # ★ description 是 ReAct 里模型选工具的唯一依据，Day32 会针对它做优化。
    #   写法要点：说清「能做什么」+「输入长什么样」+「什么时候不要用」，
    #   并给 2~3 个 example——3B 模型对 example 的敏感度远高于抽象描述。
    description: str = (
        "计算数学表达式，返回精确结果。当问题涉及任何加减乘除、乘方、开方、"
        "取整或数值大小比较时，必须使用本工具，不要自己心算。\n"
        "输入：一个纯数字表达式字符串，只能包含数字、运算符 + - * / % ** 、"
        "括号和比较符 > < >= <= ==，不能包含中文、单位、货币符号或等号。\n"
        "示例：\n"
        "  输入 123 * 456        输出 56088\n"
        "  输入 899 + 68         输出 967\n"
        "  输入 (1299+120) > 1000  输出 true\n"
        "不适用：查询商品价格、查找资料（那是 knowledge_search 的职责）。"
    )
    args_schema: Type[BaseModel] = CalculatorInput

    # 供 Day32 错误分析使用的运行期记录（ClassVar 以免被 pydantic 当成字段）
    call_log: ClassVar[list] = []

    def _run(self, expression: str, run_manager: Any = None) -> str:
        normalized = normalize_expression(expression)
        record = {"raw": expression, "normalized": normalized}
        try:
            value = safe_eval(expression)
            out = format_number(value)
            record.update(ok=True, result=out)
            return out
        except UnsafeExpression as exc:
            # 报错信息要「可自愈」：告诉模型错在哪、应该怎么改，
            # 否则 3B 模型会拿着同样的错误输入原地重试直到死循环（Day32 重点）。
            msg = (f"ERROR: {exc} | 你输入的是：{expression!r}。"
                   f"请只输入纯数字表达式，例如 899 + 68。")
            record.update(ok=False, result=msg)
            return msg
        finally:
            type(self).call_log.append(record)

    async def _arun(self, expression: str, run_manager: Any = None) -> str:
        return self._run(expression)
