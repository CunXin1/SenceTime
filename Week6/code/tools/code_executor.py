"""
code_executor.py — Week6 Day30.1
CodeExecutorTool：调用 Python `ast` 模块做语法检查与风险扫描，**不真正执行代码**。
CodeExecutorTool: static syntax check + risk scan via `ast`. It never executes code.

★ 名字叫 Executor，实际不 execute —— 这是任务书的明确要求（30.1「非真正执行，
  保证安全」）。工具的 description 里必须把这一点写死，否则模型会拿它当解释器用，
  问出「帮我运行这段代码看输出是多少」，然后把工具返回的「语法正确」误读成
  「运行结果」——这是 Day32 记录到的一类真实失败模式。

★ 为什么单有 ast.parse 不够
    `ast.parse` 只做**语法**分析，不做语义分析。下面这段能通过 ast.parse：
        def f():
            return undefined_variable + 1
    未定义变量、类型错误、缩进无关的逻辑错误，它一概查不出。所以本工具明确
    只承诺「语法层面」的结论，并额外补两件 ast 能做到的事：

    1) **危险节点扫描**：遍历语法树找 import os/sys/subprocess、eval/exec/compile/
       __import__/open 调用、以及 `__class__`/`__bases__`/`__subclasses__`/`__globals__`
       这类沙箱逃逸跳板。这是静态可判定的，属于 ast 的能力范围。
    2) **结构统计**：函数数、类数、循环数、最大嵌套深度。给模型一个可以写进
       Final Answer 的、有信息量的结论，而不是干巴巴一句「语法正确」。

★ 与 CalculatorTool 的白名单机制的区别
    Calculator 是**求值**，所以必须白名单（不在白名单里的一律拒绝执行）。
    CodeExecutor 是**审查**，从不求值，所以可以用黑名单（列出已知危险模式并报告）。
    黑名单在「审查并告警」场景是合适的——漏报只是少一条告警，不会导致代码被执行，
    因为这里**根本没有执行路径**。

用法 / Usage:
    from tools.code_executor import CodeExecutorTool
    tool = CodeExecutorTool()
    tool.run("def f(x):\\n    return x * 2")     # -> '语法正确。...'
    tool.run("def f(:")                          # -> '语法错误：第 1 行 ...'
    tool.run("import os\\nos.system('rm -rf /')") # -> '语法正确，但发现 2 处风险：...'
"""

from __future__ import annotations

import ast
from typing import Any, ClassVar, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# 危险模块：导入即告警（静态可判定）
_RISKY_MODULES = {
    "os": "可执行系统命令、读写任意文件",
    "sys": "可操作解释器状态与退出",
    "subprocess": "可派生任意进程",
    "shutil": "可递归删除目录",
    "socket": "可发起网络连接",
    "ctypes": "可调用任意本机函数",
    "pickle": "反序列化可导致任意代码执行",
    "importlib": "可动态导入任意模块",
}
# 危险内建函数：调用即告警
_RISKY_CALLS = {
    "eval": "执行任意表达式",
    "exec": "执行任意语句",
    "compile": "编译任意代码",
    "__import__": "动态导入，常用于绕过 import 检查",
    "open": "读写任意文件",
    "input": "阻塞等待标准输入",
    "breakpoint": "进入交互式调试器",
}
# 沙箱逃逸跳板属性
_RISKY_ATTRS = {
    "__class__": "类型内省，逃逸链起点",
    "__bases__": "基类链，常用于走到 object",
    "__subclasses__": "枚举所有子类，经典逃逸手法",
    "__globals__": "访问函数全局命名空间",
    "__builtins__": "直接访问内建命名空间",
    "__code__": "访问字节码对象",
}

_MAX_CODE_LEN = 8000


class _Scanner(ast.NodeVisitor):
    """遍历语法树，收集风险点与结构统计。不求值。"""

    def __init__(self):
        self.risks: list[str] = []
        self.funcs = self.classes = self.loops = 0
        self.max_depth = 0
        self._depth = 0

    # --- 结构统计 ---
    def visit_FunctionDef(self, node):          # noqa: N802
        self.funcs += 1
        self._descend(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):             # noqa: N802
        self.classes += 1
        self._descend(node)

    def visit_For(self, node):                  # noqa: N802
        self.loops += 1
        self._descend(node)

    visit_AsyncFor = visit_For

    def visit_While(self, node):                # noqa: N802
        self.loops += 1
        self._descend(node)

    def _descend(self, node):
        self._depth += 1
        self.max_depth = max(self.max_depth, self._depth)
        self.generic_visit(node)
        self._depth -= 1

    # --- 风险扫描 ---
    def visit_Import(self, node):               # noqa: N802
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in _RISKY_MODULES:
                self.risks.append(
                    f"第 {node.lineno} 行 import {alias.name}：{_RISKY_MODULES[top]}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):           # noqa: N802
        top = (node.module or "").split(".")[0]
        if top in _RISKY_MODULES:
            self.risks.append(
                f"第 {node.lineno} 行 from {node.module} import ...：{_RISKY_MODULES[top]}")
        self.generic_visit(node)

    def visit_Call(self, node):                 # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id in _RISKY_CALLS:
            self.risks.append(
                f"第 {node.lineno} 行调用 {node.func.id}()：{_RISKY_CALLS[node.func.id]}")
        self.generic_visit(node)

    def visit_Attribute(self, node):            # noqa: N802
        if node.attr in _RISKY_ATTRS:
            self.risks.append(
                f"第 {node.lineno} 行访问 {node.attr}：{_RISKY_ATTRS[node.attr]}")
        self.generic_visit(node)


def check_code(code: str) -> str:
    """对外的纯函数入口：语法检查 + 风险扫描 + 结构统计。返回给模型读的自然语言。"""
    code = str(code)
    # 模型常把代码包在 ``` 里
    stripped = code.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines)

    # ★ 反转义字面量 \n / \t（Day32 修复，根因见下）
    #   ReAct 的 Action Input 是**单行**的——模型无法在其中输出真正的换行，只能写
    #   转义形式 `\n`。若工具不认，多行代码一律被判语法错，模型拿着同一份输入
    #   反复重试直到撞步数上限（Day32 实测：S5 连续 6 次相同调用，死循环）。
    #   这与 CalculatorTool 磨平全角字符是同一类「输入归一化」：把模型受格式所限
    #   而必然产生的写法，还原成工具真正需要的形式。
    #   保守条件：仅当串里没有真换行时才反转义，避免误伤本就正确的多行输入。
    if "\\n" in code and "\n" not in code:
        code = code.replace("\\n", "\n").replace("\\t", "\t")

    if not code.strip():
        return "ERROR: 代码为空，请提供要检查的 Python 代码。"
    if len(code) > _MAX_CODE_LEN:
        return f"ERROR: 代码过长（{len(code)} 字符 > {_MAX_CODE_LEN}），已拒绝。"

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        # 语法错误要给出行/列/原因，模型才能定位并改对；只说「语法错误」会导致
        # 它盲改后重试，进而死循环（Day32 记录的失败模式之一）。
        loc = f"第 {exc.lineno} 行"
        if exc.offset:
            loc += f"第 {exc.offset} 列"
        return (f"语法错误：{loc}：{exc.msg}。"
                f"出错代码：{(exc.text or '').strip()!r}。请修正后重新检查。")
    except (ValueError, MemoryError, RecursionError) as exc:
        return f"ERROR: 无法解析代码（{type(exc).__name__}: {exc}）。"

    scanner = _Scanner()
    scanner.visit(tree)

    parts = [f"语法正确（共 {len(code.splitlines())} 行）。"]
    parts.append(
        f"结构：{scanner.funcs} 个函数、{scanner.classes} 个类、"
        f"{scanner.loops} 个循环，最大嵌套深度 {scanner.max_depth}。")

    if scanner.risks:
        uniq = list(dict.fromkeys(scanner.risks))
        parts.append(f"⚠️ 发现 {len(uniq)} 处安全风险：" + "；".join(uniq) + "。")
    else:
        parts.append("未发现危险导入或危险调用。")

    # 边界必须说清楚，否则模型会把「语法正确」当成「代码没问题/能跑出正确结果」
    parts.append("注意：本检查为静态语法分析，代码未被执行，"
                 "因此不能发现未定义变量、类型错误等运行时问题。")
    return "".join(parts)


class CodeExecutorInput(BaseModel):
    code: str = Field(
        description="要做语法检查的 Python 代码片段，可以是多行。不要带 ``` 代码块标记。"
    )


class CodeExecutorTool(BaseTool):
    """LangChain 工具：Python 代码静态语法检查（不执行）。"""

    name: str = "code_check"
    # ★ 工具名故意叫 code_check 而不是 code_executor：ReAct 里模型极大程度上
    #   按「名字」的字面意思选工具，叫 executor 会诱导它拿来求运行结果。
    #   Day32 会用 A/B 对照验证这个改名的收益。
    description: str = (
        "检查一段 Python 代码的语法是否正确，并扫描其中的危险操作"
        "（如 import os、eval、exec、open 等）。\n"
        "输入：Python 代码字符串。\n"
        "输出：语法是否正确、出错位置、代码结构统计、安全风险列表。\n"
        "重要：本工具只做静态分析，**不会运行代码，也不会返回代码的运行结果**。"
        "如果用户想知道某个算式的计算结果，请改用 calculator，不要用本工具。"
    )
    args_schema: Type[BaseModel] = CodeExecutorInput

    call_log: ClassVar[list] = []

    def _run(self, code: str, run_manager: Any = None) -> str:
        out = check_code(code)
        type(self).call_log.append({"code": code[:200], "result": out})
        return out

    async def _arun(self, code: str, run_manager: Any = None) -> str:
        return self._run(code)
