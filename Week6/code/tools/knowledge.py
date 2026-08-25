"""
knowledge.py — Week6 Day28.3
KnowledgeRetrievalTool：基于本地 JSON 文件的模拟知识库检索。
KnowledgeRetrievalTool: retrieval over a local JSON knowledge base.

★ 为什么不用向量检索 / why no embeddings
    任务书要求「基于本地 JSON 文件的模拟知识库」。上向量库（FAISS/Chroma + embedding
    模型）会引入额外依赖与一个几百 MB 的 embedding 模型，且在 6 条商品 + 5 条文档
    这种规模上，稠密检索相对关键词检索没有任何优势，反而引入「检索不确定性」——
    Day32 要做错误归因，如果检索层本身是个黑盒，就分不清「模型选错工具」还是
    「工具召回错文档」。故采用**确定性的、可解释的**混合打分：

        score = 4.0 * 关键词精确命中
              + 2.0 * 标题子串命中
              + 1.0 * 正文 bigram 重合度
              + 0.5 * 类型词命中（价格/运费/保修…）

    每次检索都把命中理由写进 last_hits，Day32 可直接引用。

★ 中文分词的处理
    中文没有空格，按空格切词会得到一整句。这里不引入 jieba（又一个依赖，且对
    「星尘X1」这种生造词表现不稳），改用 **字符 bigram**：把查询和正文都切成
    相邻二字组合求重合度。对中文短查询，bigram 的召回质量已经足够，且完全确定性。

★ 返回格式为什么是纯文本而非 JSON
    返回值会作为 ReAct 的 Observation 直接拼进 prompt。3B 模型读 JSON 时容易把
    花括号和引号当成要继续输出的格式，进而破坏 Thought/Action 结构（Day29 实测）。
    返回自然语言句子，模型的抽取准确率明显更高。

用法 / Usage:
    from tools.knowledge import KnowledgeRetrievalTool
    tool = KnowledgeRetrievalTool()
    tool.run("星尘X1智能音箱的价格")   # -> '【星尘 X1 智能音箱】售价 899 元，运费 68 元...'
    tool.run("退货")                   # -> 退换货政策条目
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KB = ROOT / "Week6" / "data" / "knowledge_base.json"

TOP_K = 2                # 默认返回条数：多了会把 Observation 撑长、稀释关键数字
MIN_SCORE = 1.0          # 低于此分视为未命中，宁可明确说「没找到」也不要糊一个近似答案

# 「意图词」——出现这些词说明用户在问某个具体字段，用于给含该字段的条目加分。
_FIELD_HINTS = {
    "价格": ["价格", "售价", "多少钱", "价钱", "报价"],
    "运费": ["运费", "邮费", "配送费", "包邮"],
    "库存": ["库存", "有货", "还有"],
    "保修": ["保修", "质保", "保修期"],
}


def _normalize(text: str) -> str:
    """统一大小写与全角，去掉标点，便于匹配。"""
    s = str(text).lower().strip()
    s = s.translate(str.maketrans({
        "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
        "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
        "（": "(", "）": ")", "，": ",", "　": " ",
    }))
    # 保留中日韩、字母、数字，其余替换为空格
    s = re.sub(r"[^\w一-鿿]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _bigrams(text: str) -> set:
    """字符 bigram。中文无空格，bigram 比整句匹配召回好、比单字精确。"""
    s = _normalize(text).replace(" ", "")
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}


class KnowledgeBase:
    """加载并检索本地 JSON 知识库。与 LangChain 解耦，便于单测。"""

    def __init__(self, path: Path | str = DEFAULT_KB):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"知识库不存在：{self.path}")
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.entries = data["entries"]
        self.meta = data.get("_meta", {})

    def search(self, query: str, top_k: int = TOP_K) -> list:
        """返回 [(score, entry, reasons), ...]，按分数降序。"""
        q_norm = _normalize(query)
        q_compact = q_norm.replace(" ", "")
        q_bi = _bigrams(query)

        scored = []
        for entry in self.entries:
            score, reasons = 0.0, []

            # ① 关键词精确命中（权重最高，且这是知识库作者显式声明的检索入口）
            for kw in entry.get("keywords", []):
                if _normalize(kw).replace(" ", "") in q_compact:
                    score += 4.0
                    reasons.append(f"关键词'{kw}'")

            # ② 标题子串命中
            title_compact = _normalize(entry["title"]).replace(" ", "")
            if title_compact and title_compact in q_compact:
                score += 2.0
                reasons.append("标题完整命中")

            # ③ 正文 bigram 重合度（归一化到 [0,1] 再乘权重）
            e_bi = _bigrams(entry["title"] + entry["content"])
            if q_bi and e_bi:
                overlap = len(q_bi & e_bi) / len(q_bi)
                if overlap > 0:
                    score += overlap
                    reasons.append(f"正文重合{overlap:.0%}")

            # ④ 字段意图词：问「价格」时，有「价格」字段的商品条目更相关
            for field, hints in _FIELD_HINTS.items():
                if any(h in q_compact for h in hints) and field in entry.get("fields", {}):
                    score += 0.5
                    reasons.append(f"含'{field}'字段")

            if score >= MIN_SCORE:
                scored.append((score, entry, reasons))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


def _render(entry: dict) -> str:
    """把条目渲染成一句自然语言，供 LLM 直接抽取（不返回 JSON，见文件抬头说明）。"""
    return f"【{entry['title']}】{entry['content']}"


class KnowledgeInput(BaseModel):
    query: str = Field(
        description="要查询的商品名称或问题关键词，例如 星尘X1 价格、退货政策、运费规则"
    )


class KnowledgeRetrievalTool(BaseTool):
    """LangChain 工具：检索本地商品/政策知识库。"""

    name: str = "knowledge_search"
    # ★ description 的边界写得越清楚，Day32 的「选错工具」率越低。
    #   特别要写明「本工具不做计算」——否则模型会把 "899+68" 也丢给它。
    description: str = (
        "查询商品信息（价格、运费、库存、保修）和店铺政策（退换货、配送、"
        "会员折扣、发票）。当问题涉及某个具体商品或店铺规则时，必须先用本工具查到"
        "事实，不要凭记忆编造价格。\n"
        "输入：商品名或问题关键词，例如 星尘X1、追光S3 运费、退货政策。\n"
        "输出：匹配到的商品/政策原文。\n"
        "不适用：本工具只负责「查」，不做任何算术。查到数字后如需加减乘除或比较大小，"
        "请改用 calculator。"
    )
    args_schema: Type[BaseModel] = KnowledgeInput

    _kb: KnowledgeBase = PrivateAttr()
    call_log: ClassVar[list] = []

    def __init__(self, kb_path: Path | str = DEFAULT_KB, **kwargs):
        super().__init__(**kwargs)
        self._kb = KnowledgeBase(kb_path)

    def _run(self, query: str, run_manager: Any = None) -> str:
        hits = self._kb.search(query)
        record = {
            "query": query,
            "hits": [{"id": e["id"], "score": round(s, 2), "reasons": r}
                     for s, e, r in hits],
        }
        type(self).call_log.append(record)

        if not hits:
            # 明确说「没找到」+ 给出可用范围，让模型能自我纠正查询词，
            # 而不是拿着同一个查询反复重试（Day32 死循环的主要来源之一）。
            titles = "、".join(e["title"] for e in self._kb.entries
                              if e["type"] == "product")
            return (f"未找到与「{query}」相关的信息。"
                    f"知识库中的商品有：{titles}。"
                    f"也可查询：运费规则、退换货政策、保修服务、会员折扣、发票说明。"
                    f"请换一个更准确的关键词重试，不要编造答案。")

        return "\n".join(_render(e) for _, e, _ in hits)

    async def _arun(self, query: str, run_manager: Any = None) -> str:
        return self._run(query)
