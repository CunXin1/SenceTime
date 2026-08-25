"""
local_llm.py — Week6 Day29
把本地 Qwen2.5-3B（HF transformers）包成 LangChain 的 LLM，供 create_react_agent 使用。
Wrap the local HF model as a LangChain LLM for create_react_agent.

★ 为什么不用 HuggingFacePipeline
    文本版 ReAct 有两个硬要求，现成的 HuggingFacePipeline 都不好满足：

    1) **必须能在 "Observation:" 处停住**。ReAct 的循环是：模型写 Thought/Action/
       Action Input → 框架真正去调工具 → 把结果作为 Observation 回填。如果模型不停，
       它会自己把 Observation 也编出来（幻觉工具结果），整个 Agent 就退化成自问自答。
       HuggingFacePipeline 对 stop 序列的支持依赖版本且不稳，这里用 StoppingCriteria
       精确控制。

    2) **必须能让 chat 模型「续写」assistant 轮**。这是本文件最关键的设计：
       Qwen 是 chat 模型，apply_chat_template 后每个 assistant 轮都从空开始。
       但 ReAct 的 agent_scratchpad 累积的是「模型已经说过的话」
       （Thought1/Action1/Observation1/Thought2...），模型必须**接着往下写**，
       而不是重新开一轮。
       做法：apply_chat_template(add_generation_prompt=True) 之后，把 scratchpad
       直接拼在生成起点后面作为 **assistant 前缀填充（prefill）**。这样模型看到的是
       一个「说到一半的自己」，自然会续写下一个 Thought。
       ——如果不这么做，而是把 scratchpad 塞进 user 轮，3B 模型会把它当成用户提供的
       资料、然后从头再输出一遍 Thought1，直接死循环（Day32 实测过这个失败模式）。

★ 贪心解码
    do_sample=False。Agent 的评测要可复现；采样会让同一道题两次跑出不同的工具序列，
    Day32 的失败模式统计就失去意义。这一点与 Week3/Week4 的评测口径保持一致。

用法 / Usage:
    from local_llm import LocalQwenLLM
    llm = LocalQwenLLM.build("models/Qwen2.5-3B-week4-dpo-merged")
    llm.invoke("SYSTEM_MARKER...")     # 一般不直接调，交给 AgentExecutor
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import PrivateAttr

ROOT = Path(__file__).resolve().parents[2]

# 提示词里用这三个哨兵切分 system / user / assistant-prefill 三段。
# 之所以要哨兵：LangChain 的 create_react_agent 只接受**一个**字符串模板
# （含 {input} 与 {agent_scratchpad}），而我们需要把它还原成 chat 三段式。
SYS_MARK = "<<<SYSTEM>>>"
USER_MARK = "<<<USER>>>"
ASSIST_MARK = "<<<ASSISTANT>>>"

# ReAct 的停止串。写两个变体是因为模型有时不带前导换行。
STOP_STRINGS = ["\nObservation:", "Observation:", "\nObservation：", "Observation："]


def split_prompt(text: str) -> tuple[str, str, str]:
    """把带哨兵的单串提示还原成 (system, user, assistant_prefill)。"""
    sys_part = user_part = assist_part = ""
    if SYS_MARK in text:
        _, rest = text.split(SYS_MARK, 1)
        if USER_MARK in rest:
            sys_part, rest = rest.split(USER_MARK, 1)
        if ASSIST_MARK in rest:
            user_part, assist_part = rest.split(ASSIST_MARK, 1)
        else:
            user_part = rest
    else:                       # 没有哨兵就整串当 user，保证不会炸
        user_part = text
    return sys_part.strip(), user_part.strip(), assist_part.lstrip("\n")


class LocalQwenLLM(LLM):
    """本地 Qwen chat 模型的 LangChain 封装，支持 assistant 前缀续写与停止串。"""

    model_path: str
    max_new_tokens: int = 512
    verbose_io: bool = False

    _model: Any = PrivateAttr(default=None)
    _tok: Any = PrivateAttr(default=None)

    # 全周共享的调用统计，Day32 分析用
    call_stats: ClassVar[dict] = {"calls": 0, "gen_tokens": 0, "seconds": 0.0}

    @property
    def _llm_type(self) -> str:
        return "local-qwen-react"

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, model_path: str | Path, adapter: str | Path | None = None,
              max_new_tokens: int = 512, verbose_io: bool = False) -> "LocalQwenLLM":
        """加载模型（可选挂 LoRA adapter），返回可直接给 Agent 用的 LLM。"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        mp = Path(model_path)
        if not mp.is_absolute():
            mp = ROOT / mp
        if not mp.exists():
            raise FileNotFoundError(
                f"模型不存在：{mp}\n"
                f"若是 week4-dpo-merged，请先跑 Week6/code/rebuild_base_model.ps1 重建。")

        obj = cls(model_path=str(mp), max_new_tokens=max_new_tokens,
                  verbose_io=verbose_io)
        tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            mp, torch_dtype=torch.bfloat16, device_map="cuda:0",
            trust_remote_code=True)

        if adapter:
            ap = Path(adapter)
            if not ap.is_absolute():
                ap = ROOT / ap
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(ap))
            model = model.merge_and_unload()   # 合并进主干，推理更快且避免多 adapter 告警

        model.eval()
        obj._tok, obj._model = tok, model
        return obj

    # ----------------------------------------------------------------- call
    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any) -> str:
        import torch

        sys_part, user_part, assist_prefill = split_prompt(prompt)
        messages = []
        if sys_part:
            messages.append({"role": "system", "content": sys_part})
        messages.append({"role": "user", "content": user_part})

        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        # ★ 关键一步：把 scratchpad 拼成 assistant 的前缀，让模型「续写自己」。
        text += assist_prefill

        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        stop_list = list(STOP_STRINGS) + list(stop or [])
        criteria = _build_stopping_criteria(
            self._tok, inputs["input_ids"].shape[1], stop_list)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,                       # 贪心，保证可复现
                stopping_criteria=criteria,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        completion = self._tok.decode(gen_ids, skip_special_tokens=True)

        # StoppingCriteria 是按 token 判定的，会多吐出停止串本身，这里按字符串裁干净。
        for s in stop_list:
            idx = completion.find(s)
            if idx != -1:
                completion = completion[:idx]

        type(self).call_stats["calls"] += 1
        type(self).call_stats["gen_tokens"] += int(gen_ids.shape[0])
        type(self).call_stats["seconds"] += time.perf_counter() - t0

        if self.verbose_io:
            print(f"\n--- LLM 续写 ---\n{completion}\n---------------")
        return completion


def _build_stopping_criteria(tok, prompt_len: int, stops: List[str]):
    """按「已生成文本是否含停止串」判停。

    不用 token id 匹配：'Observation' 在不同上下文会被切成不同的 token 组合
    （前导空格/换行都会改变切分），按 id 匹配漏判率很高。解码成字符串再找子串
    虽然每步多一次 decode，但 3B 模型每步生成才几十 token，开销可忽略。
    """
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopOnStrings(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw) -> bool:
            text = tok.decode(input_ids[0][prompt_len:], skip_special_tokens=True)
            return any(s in text for s in stops)

    return StoppingCriteriaList([_StopOnStrings()])
