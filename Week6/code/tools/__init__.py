"""
Week6 工具包 / Week6 tool package。

Day28：CalculatorTool（AST 白名单沙箱计算）、KnowledgeRetrievalTool（本地 JSON 检索）
Day30：CodeExecutorTool（AST 语法检查 + 危险节点扫描，不真正执行）

三个工具统一走 LangChain BaseTool 接口，便于 Day29 的 create_react_agent 直接绑定。
"""

from .calculator import CalculatorTool, safe_eval, UnsafeExpression
from .knowledge import KnowledgeRetrievalTool, KnowledgeBase

__all__ = [
    "CalculatorTool",
    "KnowledgeRetrievalTool",
    "KnowledgeBase",
    "safe_eval",
    "UnsafeExpression",
]


def build_all_tools(kb_path=None):
    """构造全部工具实例。Day29/Day30 统一从这里取，保证工具集一致。

    Day30 的 CodeExecutorTool 在其文件就位后会自动加入（延迟导入，
    使得 Day28 阶段单独跑测试时不会因为文件不存在而报错）。
    """
    tools = [
        CalculatorTool(),
        KnowledgeRetrievalTool(kb_path) if kb_path else KnowledgeRetrievalTool(),
    ]
    try:
        from .code_executor import CodeExecutorTool
        tools.append(CodeExecutorTool())
    except ImportError:
        pass
    return tools
