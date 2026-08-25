# -*- coding: utf-8 -*-
"""
distill_kd.py — Week8 Day42
白盒 logit-level 知识蒸馏：Qwen2.5-3B-week4-dpo（教师）→ Qwen2.5-0.5B-Instruct（学生）。
White-box logit-level KD: Week4 DPO 3B teacher -> Qwen2.5-0.5B-Instruct student.

同一个脚本同时承担 B/C 两组（靠 YAML 里的 alpha 切换），这是刻意的：
    · alpha = 0  → 纯 SFT 对照组（B），教师根本不加载
    · alpha > 0  → KD 蒸馏组（C）
一份代码两组实验，保证 B/C 之间**除了 KD 损失项以外没有任何差异**（同数据、同超参、
同随机种子、同 collator、同学习率调度）。如果 B/C 用两个脚本，"收益到底来自蒸馏还是
来自实现差异"就永远说不清。
One script serves both arms; alpha=0 disables the teacher entirely, so B and C
differ ONLY in the KD term.

═══════════════════════════════ 42.1 原理落到代码 ═══════════════════════════════

★ 软标签（soft targets）为什么比硬标签信息量大
    硬标签是 one-hot：第 t 步的答案是 token "3"，其余 151935 个词概率为 0。
    教师的软标签是一整个分布：也许 P("3")=0.6, P("三")=0.2, P("4")=0.05 ……
    后面这些非零项 Hinton 称之为 "dark knowledge" —— 它编码了**类别之间的相似结构**
    （"3" 和 "三" 语义相近、和 "4" 是近邻数字、和 "猫" 毫无关系）。one-hot 把这些
    全抹平了。每个 token 位置从"1 bit 的监督"变成"151936 维分布的监督"，
    这就是为什么 KD 在小数据量下往往比纯 SFT 更省样本。

★ 温度系数 T 在做什么
    softmax(z/T)：T>1 把 logits 压扁，放大小概率项的相对权重，让 dark knowledge
    真正参与到损失里；T→0 退化成 one-hot（等于没蒸馏），T→∞ 退化成均匀分布（噪声）。
    本任务取 T=2.0：Qwen 的输出分布本身相当尖锐（top-1 概率常 >0.9），T=1 时 KL 几乎
    只被 top-1 支配，等价于软化版的 CE；T=2 能把 top-5~top-50 的结构带进来，
    又不至于把长尾噪声放大到主导损失。

★ 为什么 KL 项要乘 T²  —— 本任务里最容易写错的一行
    对 z_s 求导：∂/∂z_s [ KL(softmax(z_t/T) ‖ softmax(z_s/T)) ]
                = (1/T) · (p_s^{(T)} - p_t^{(T)})
    也就是**梯度幅度按 1/T 缩**，而在小 logits 极限下展开 softmax，
    (p_s-p_t) 本身又正比于 (z_s-z_t)/T，两处合起来是 1/T²。
    若不乘回 T²，把 T 从 1 调到 4 会让 KD 梯度悄悄缩小约 16 倍 —— 于是"α 的含义"
    随 T 漂移，α=0.5 在 T=1 和 T=4 下根本不是同一个权重，超参完全不可比。
    乘上 T² 之后 KD 项的梯度尺度与 CE 项同阶，α 才真的是"两个损失的相对权重"
    这个语义。
    (The T^2 factor cancels the 1/T^2 gradient shrinkage so that alpha keeps a
     T-independent meaning; without it, tuning T silently retunes alpha.)

★ 为什么必须只在 assistant 回复 token 上算损失
    训练样本的完整序列是 [system][user 指令][assistant 回复]。prompt 部分是**输入**，
    不是要学的目标。若不 mask：
      1. CE 项会让学生去"预测用户下一句会说什么" —— 学的是复述用户输入，
         生成时表现为鹦鹉学舌、重复提问。
      2. KD 项更微妙但同样有害：prompt 段占 token 总数的一大半，教师在这段上的
         分布是"如何续写用户的话"，与"如何回答"是两回事。不 mask 等于让一半以上的
         蒸馏信号去对齐一个我们根本不关心的任务，真正有用的回复段被稀释。
    实现上：labels 里 prompt 位置填 IGNORE_INDEX=-100；KL 也用同一个 mask
    （见 _kd_loss 里的 valid 选择），两个损失项作用在**完全相同的 token 集合**上，
    α 才是干净的加权。
    (Prompt tokens are masked in BOTH terms — otherwise >50% of the distillation
     signal aligns a task we do not care about.)

═══════════════════════════════ 工程取舍 ═══════════════════════════════

★ 取舍 1：自行实现，而不是用 LLaMA-Factory 的"蒸馏功能"
    任务书写的是"利用 LLaMA-Factory 的蒸馏功能（或自行实现）"。本地 LF 0.9.6.dev0
    源码核查结论：**没有蒸馏功能**，举证三条（详见 Week8/docs/Day42_知识蒸馏.md）：
      a) grep -rniI "teacher" LLaMA-Factory/src/ --include=*.py  →  0 命中。
         没有 teacher 概念，就不可能有 teacher-student 蒸馏。
      b) finetuning_args.py:460 的 stage 枚举是
         Literal["pt","sft","rm","ppo","dpo","kto"] —— 没有 kd / distill。
      c) 唯一的 KL 散度实现 trainer_utils.py:743 _kl_divergence 只被
         _asft_cross_entropy（ASFT，一种 DPO 家族的对齐损失）调用，比的是
         policy vs **同尺寸 reference model**，用途是防漂移，不是跨尺寸蒸馏。
    与 Week7 "LF 的 --quantization_method awq 其实导不出 AWQ" 是同一类情况：
    任务书对 LF 能力的描述再次与源码不符，因此走"自行实现"。
    (LF 0.9.6.dev0 has no KD trainer — zero "teacher" hits, no kd stage, and its
     only KL is ASFT's same-size reference regulariser.)

★ 取舍 2：白盒 logit KD，而不是黑盒序列级蒸馏
    黑盒方案（教师生成回答 → 学生 SFT）不需要词表一致，但每个 token 只拿回 1 bit
    监督，且完全用不到"软标签/温度系数"这两个 42.1 明确要求的概念。
    本机核实（见 docs）：教师与学生 get_vocab() 完全相等、样本 encode 结果逐 id
    相同、chat_template 渲染结果相同（vocab_size 都是 151936）。
    词表对齐 ⇒ 两个模型在同一个 token 位置输出的 logits 落在**同一个 151936 维
    坐标系**里，可以直接逐位置算 KL。这是能做真蒸馏的前提，也是本任务最关键的判断。
    (Identical vocab => logits live in the same 151936-d space => real logit KD.)

★ 取舍 3：学生全参训练，不用 LoRA
    KD 的本质是**重塑学生的整个输出分布**。0.5B 模型本身只有 494M 参数，
    再套一个 rank-32 LoRA（约 8M 可训参数，1.6%）等于让蒸馏信号只能穿过一个
    极窄的瓶颈，教师分布里的结构大部分没地方存。前几周用 LoRA 是为了 3B 省显存，
    0.5B 全参在 24GB 上完全跑得动（实测见日志 peak VRAM），没有理由自我限制。
    B 组同样全参，保证对照公平。
    (Full-parameter: a rank-32 adapter on a 0.5B student is too narrow a channel
     for distribution-level supervision; and 0.5B full FT fits in 24GB anyway.)

★ 取舍 4：在线教师前向，不做离线 top-K logits 预计算
    离线预计算能省教师那 6.2GB 显存，但代价是磁盘：4684 条 × 768 token × top-64
    × (int32 id + fp16 prob) ≈ 2.2GB，而 C 盘只剩 35GB，且 top-K 截断会丢掉
    长尾的 dark knowledge（截断后还要重归一化，KL 的语义就变了）。
    实测在线方案峰值显存有充足余量（见 run_meta.json 的 peak_vram_mib），
    没必要为省显存牺牲精度和磁盘。
    (Online teacher forward: offline top-K would cost ~2.2GB disk and truncate the
     very long tail that carries the dark knowledge. VRAM headroom made it moot.)

★ 取舍 5：forward KL（mode-covering），不是 reverse KL
    用 KL(p_teacher ‖ p_student)：教师概率高的地方学生必须也给概率，否则罚很重
    → 学生倾向"覆盖"教师的所有模式，输出更多样但可能更平。
    reverse KL(p_student ‖ p_teacher) 是 mode-seeking，学生会塌到教师的单一主峰，
    生成更确定但更保守。经典 KD（Hinton）用 forward，本任务遵循经典形式以对齐 42.1
    的描述；F.kl_div(input=log p_s, target=p_t) 算的正是 forward KL。

★ 取舍 6：KL 只在有效位置上算（valid 索引选择）
    logits 是 [B, L, 151936] 的巨型张量，B=2/L=768 时单个就有 467MB(bf16)，
    fp32 log_softmax 中间量再翻倍。先按 labels != -100 把有效位置 gather 出来
    （通常只占 40~60%），再做 softmax/KL，省掉近一半的 softmax 中间显存，
    同时天然实现了"只在 assistant 回复上蒸馏"。

用法 / Usage（仓库根目录 / from repo root）:
    # 冒烟：只跑 4 个优化步，确认 loss 能出、显存够
    .venv/Scripts/python.exe Week8/scripts/distill_kd.py --config Week8/configs/distill_kd.yaml --smoke
    # 正式（C 组：KD）
    .venv/Scripts/python.exe Week8/scripts/distill_kd.py --config Week8/configs/distill_kd.yaml
    # 正式（B 组：纯 SFT 对照，alpha=0，不加载教师）
    .venv/Scripts/python.exe Week8/scripts/distill_kd.py --config Week8/configs/student_sft_baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[2]
IGNORE_INDEX = -100  # 与 transformers / LF 保持一致


# ─────────────────────────────────────────────────────────────────────────────
# 显存监控（沿用 Week3 run_experiments.py / Week4 run_dpo.py 的 VramMonitor 模式）
# ─────────────────────────────────────────────────────────────────────────────


class VramMonitor:
    """后台轮询 nvidia-smi 记录整卡峰值显存（MiB）。

    ★ 为什么不用 torch.cuda.max_memory_allocated()：那个只算本进程 PyTorch
    allocator 的量，不含 CUDA context、cuBLAS workspace，也不含别的进程。
    对外报"这条路线需要多大的卡"时，整卡占用才是有意义的数字。
    Whole-device peak is the number that answers "what GPU do I need"."""

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _query(self) -> int:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            return max(int(x) for x in out.stdout.split())
        except Exception:
            return 0

    def _loop(self):
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, self._query())
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
# 数据集：alpaca → chat template → input_ids / labels（prompt 段 mask 掉）
# ─────────────────────────────────────────────────────────────────────────────


class AlpacaSFTDataset(Dataset):
    """把 Week2 清洗好的 alpaca_clean.json 编码成 (input_ids, labels)。

    ★ 为什么自己写而不用 LF 的 dataloader：本脚本要在 compute_loss 里同时驱动
    两个模型，用 LF 的 workflow 反而要绕开它一堆 model_args 假设。自己编码
    只有 30 行，而且能把"prompt 到哪里结束"这个关键边界完全显式化 ——
    这是 KD 正确性的命门，不该藏在框架里。

    ★ prompt 边界怎么算：分两次渲染 chat template。
        prompt_ids = template([user], add_generation_prompt=True)
        full_ids   = template([user, assistant])
      前者的长度就是要 mask 的长度。这比"搜 <|im_start|>assistant 的位置"稳健，
      因为它完全跟随 tokenizer 自己的模板，模板改了也不会错位。
      (Two renders: len(prompt_ids) IS the mask length — robust to template changes.)
    """

    def __init__(self, path: Path, tokenizer, cutoff_len: int, max_samples: int | None = None):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if max_samples is not None:
            raw = raw[:max_samples]

        self.examples: list[dict] = []
        n_truncated = 0
        for rec in raw:
            instruction = (rec.get("instruction") or "").strip()
            extra = (rec.get("input") or "").strip()
            output = (rec.get("output") or "").strip()
            if not instruction or not output:
                continue
            # alpaca 的 input 字段是"补充材料"，拼到指令后面，与 LF 的 alpaca
            # 模板保持一致的语义
            user = f"{instruction}\n{extra}" if extra else instruction

            msgs_prompt = [{"role": "user", "content": user}]
            prompt_ids = tokenizer.apply_chat_template(
                msgs_prompt, tokenize=True, add_generation_prompt=True)
            msgs_full = msgs_prompt + [{"role": "assistant", "content": output}]
            full_ids = tokenizer.apply_chat_template(
                msgs_full, tokenize=True, add_generation_prompt=False)

            if len(prompt_ids) >= cutoff_len:
                continue  # prompt 自己就超长，回复一个 token 都放不下，丢掉
            if len(full_ids) > cutoff_len:
                full_ids = full_ids[:cutoff_len]
                n_truncated += 1

            labels = list(full_ids)
            labels[: len(prompt_ids)] = [IGNORE_INDEX] * len(prompt_ids)  # ★ mask prompt
            self.examples.append({"input_ids": full_ids, "labels": labels})

        n_resp = sum(sum(1 for x in e["labels"] if x != IGNORE_INDEX) for e in self.examples)
        n_tok = sum(len(e["input_ids"]) for e in self.examples)
        self.stats = {
            "n_samples": len(self.examples),
            "n_raw": len(raw),
            "n_truncated": n_truncated,
            "n_tokens": n_tok,
            "n_loss_tokens": n_resp,
            "loss_token_ratio": round(n_resp / max(n_tok, 1), 4),
        }
        print(f"[data] {len(self.examples)} 条样本（原始 {len(raw)}），截断 {n_truncated} 条")
        print(f"[data] token 总量 {n_tok}，其中参与损失的 assistant token "
              f"{n_resp} ({100 * n_resp / max(n_tok, 1):.1f}%)  ← 其余是被 mask 的 prompt")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


@dataclass
class PadCollator:
    """右填充到 batch 内最长。pad 位置的 label 也是 -100，所以不会进损失。

    ★ pad_token_id 用 <|endoftext|>(151643) 而不是 eos <|im_end|>(151645)：
      两者混用会让"回复真的结束"和"这里只是填充"在 input_ids 上无法区分。"""

    pad_token_id: int

    def __call__(self, feats: list[dict]) -> dict:
        maxlen = max(len(f["input_ids"]) for f in feats)
        ids, labels, attn = [], [], []
        for f in feats:
            n = maxlen - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_token_id] * n)
            labels.append(f["labels"] + [IGNORE_INDEX] * n)
            attn.append([1] * len(f["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# KD Trainer
# ─────────────────────────────────────────────────────────────────────────────


class KDTrainer(Trainer):
    """在标准 SFT 的 CE 之上叠加教师软标签的 KL 项。

        L = alpha * T^2 * KL(p_t^(T) ‖ p_s^(T)) + (1 - alpha) * CE(z_s, y)

    ★ 教师故意存成 self._teacher = [model]（藏在 list 里）
      Trainer / accelerate 会遍历自己持有的 nn.Module 去做 prepare、DDP 包装、
      state_dict 保存。教师是冻结的、不该被保存也不该被包装，套一层 list 让它
      对这些自动机制不可见，同时又能正常调用。
      (Teacher hidden inside a list so accelerate/Trainer never wraps or saves it.)
    """

    def __init__(self, *args, teacher=None, kd_alpha: float = 0.0,
                 kd_temperature: float = 2.0, loss_log_path: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._teacher = [teacher] if teacher is not None else []
        self.kd_alpha = float(kd_alpha)
        self.kd_temperature = float(kd_temperature)
        self.loss_log_path = loss_log_path
        # 累积各分量，用于 log 时输出（Trainer 只会 log 加权后的总 loss）
        self._acc = {"ce": 0.0, "kd": 0.0, "n": 0}

    @property
    def teacher(self):
        return self._teacher[0] if self._teacher else None

    def _kd_loss(self, s_logits: torch.Tensor, t_logits: torch.Tensor,
                 labels: torch.Tensor) -> torch.Tensor:
        """在 assistant 回复位置上算 T^2 加权的 forward KL。

        输入都是 [B, L, V] 的**未偏移** logits；这里做和 CE 一样的 shift：
        位置 i 的 logits 预测的是 token i+1，所以 logits 去掉最后一个、
        labels 去掉第一个，两边才对齐。★ 这个 shift 必须和 CE 完全一致，
        否则 KD 项和 CE 项会在错位的位置上互相打架。
        """
        s = s_logits[:, :-1, :]
        t = t_logits[:, :-1, :]
        lab = labels[:, 1:]

        valid = lab != IGNORE_INDEX          # [B, L-1] 只保留 assistant token
        if valid.sum() == 0:
            return s_logits.new_zeros(())

        # ★ 先 gather 有效位置再 softmax：把 [B,L,151936] 压成 [N,151936]，
        #   N 通常只有 B*L 的一半，省掉近一半 softmax 中间显存。
        s = s[valid]                          # [N, V]
        t = t[valid]                          # [N, V]

        T = self.kd_temperature
        # ★ fp32 计算：151936 维的 log_softmax 在 bf16 下（尾数只有 8 bit）
        #   累加误差足以让 KL 出现负值甚至 NaN。这里显式升到 fp32。
        log_p_s = F.log_softmax(s.float() / T, dim=-1)
        p_t = F.softmax(t.float() / T, dim=-1)

        # F.kl_div(input=log q, target=p) = Σ p·(log p − log q) = KL(p ‖ q)
        # 取 input=学生, target=教师 ⇒ forward KL(p_teacher ‖ p_student)，mode-covering
        kl = F.kl_div(log_p_s, p_t, reduction="none").sum(-1).mean()
        return (T ** 2) * kl                  # ★ 补回 1/T² 的梯度缩放

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"])
        s_logits = outputs.logits

        # ── 标准 CE（硬标签）。自己算而不是让 model(labels=...) 内部算，是为了
        #    确保 CE 和 KL 用的是同一个 shift、同一个 mask、同一个 reduction。
        ce = F.cross_entropy(
            s_logits[:, :-1, :].reshape(-1, s_logits.size(-1)).float(),
            labels[:, 1:].reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

        if self.kd_alpha > 0.0 and self.teacher is not None:
            with torch.no_grad():  # ★ 教师全程冻结，不建计算图
                t_logits = self.teacher(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                ).logits
            kd = self._kd_loss(s_logits, t_logits, labels)
            del t_logits  # 几百 MB 级别的张量，尽早还给 allocator
            loss = self.kd_alpha * kd + (1.0 - self.kd_alpha) * ce
        else:
            kd = s_logits.new_zeros(())
            loss = ce

        self._acc["ce"] += float(ce.detach())
        self._acc["kd"] += float(kd.detach())
        self._acc["n"] += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, start_time: float | None = None) -> None:
        """把 CE / KD 分量一起写进日志。

        ★ 只看总 loss 是看不出蒸馏在不在工作的 —— α 加权后总 loss 的变化可能来自
        任一项。分量分开记，才能在报告里说"KD 项确实在下降"。同时记 exp(CE) 作为
        训练集困惑度，它和 B 组是可直接比较的（B 组 CE 就是全部损失）。"""
        if self._acc["n"] > 0 and "loss" in logs:
            n = self._acc["n"]
            logs["ce_loss"] = round(self._acc["ce"] / n, 4)
            logs["kd_loss"] = round(self._acc["kd"] / n, 4)
            logs["ppl_train"] = round(math.exp(min(self._acc["ce"] / n, 20)), 3)
            self._acc = {"ce": 0.0, "kd": 0.0, "n": 0}
        super().log(logs, start_time)
        if self.loss_log_path is not None and self.is_world_process_zero():
            with open(self.loss_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(logs, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────


def load_cfg(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    # 相对路径统一按仓库根解析，脚本从哪个目录调都一样
    for k in ("student_model", "teacher_model", "dataset_path", "output_dir", "log_dir"):
        v = cfg.get(k)
        if v and not Path(v).is_absolute():
            cfg[k] = str((ROOT / v).resolve())
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：64 条样本 / 4 个优化步 / 不保存权重，只验证能不能跑")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    run_name = cfg["run_name"] + ("_smoke" if args.smoke else "")

    log_dir = Path(cfg["log_dir"]) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg["output_dir"])

    set_seed(int(cfg.get("seed", 42)))
    alpha = float(cfg.get("kd_alpha", 0.0))
    temperature = float(cfg.get("kd_temperature", 2.0))

    print("=" * 78)
    print(f"[run] {run_name}   alpha={alpha}  T={temperature}  "
          f"({'KD 蒸馏组' if alpha > 0 else '纯 SFT 对照组（不加载教师）'})")
    print(f"[run] student = {cfg['student_model']}")
    print(f"[run] teacher = {cfg['teacher_model'] if alpha > 0 else '(不使用)'}")
    print("=" * 78, flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["student_model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ds = AlpacaSFTDataset(
        Path(cfg["dataset_path"]), tok,
        cutoff_len=int(cfg["cutoff_len"]),
        max_samples=64 if args.smoke else args.max_samples,
    )

    # ── 学生：fp32 权重 + bf16 autocast（HF Trainer 的 bf16=True 路径）。
    #    ★ 不用 dtype=bfloat16 直接加载：那样优化器状态也是 bf16，0.5B 全参
    #      在 lr=1e-5 量级下，bf16 的 8-bit 尾数会把大部分小更新直接吃掉
    #      （更新量小于权重的 2^-8 相对精度时加不进去）。fp32 master weight
    #      + bf16 前向是稳定性/显存的合理折中。
    student = AutoModelForCausalLM.from_pretrained(
        cfg["student_model"], dtype=torch.float32, attn_implementation="sdpa")
    student.config.use_cache = False  # 与 gradient_checkpointing 互斥
    n_param = sum(p.numel() for p in student.parameters())
    print(f"[model] student 参数量 {n_param / 1e6:.1f}M  (全部可训练)")

    teacher = None
    if alpha > 0.0:
        # ★ 教师用 fp16 加载：只做前向，不需要 fp32 精度；3B fp16 约 6.2GB，
        #   fp32 就要 12.4GB，加上学生的全参优化器状态会直接爆卡。
        teacher = AutoModelForCausalLM.from_pretrained(
            cfg["teacher_model"], dtype=torch.float16, attn_implementation="sdpa")
        teacher.eval()
        teacher.requires_grad_(False)
        teacher.config.use_cache = False
        teacher.to("cuda")
        print(f"[model] teacher 参数量 "
              f"{sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M (fp16, 冻结)")

    targs = TrainingArguments(
        output_dir=str(log_dir / "checkpoints"),
        num_train_epochs=1 if args.smoke else float(cfg["num_train_epochs"]),
        max_steps=4 if args.smoke else -1,
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(cfg.get("logging_steps", 5)),
        bf16=True,
        optim=cfg.get("optim", "adamw_torch_fused"),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        # ★ Windows 铁律（Week2 起的结论）：spawn 出来的 dataloader worker 会
        #   触发 CUDA IPC 误报 OOM。必须是 0。
        dataloader_num_workers=0,
        # ★ 只留最终权重：C 盘只剩 35GB，0.5B 的中间 checkpoint 连优化器状态
        #   每个约 6GB，2 轮下来能把盘塞满。
        save_strategy="no",
        report_to=[],
        seed=int(cfg.get("seed", 42)),
        remove_unused_columns=False,  # 我们的 dataset 返回的就是模型要的字段
    )

    trainer = KDTrainer(
        model=student,
        args=targs,
        train_dataset=ds,
        data_collator=PadCollator(pad_token_id=tok.pad_token_id),
        teacher=teacher,
        kd_alpha=alpha,
        kd_temperature=temperature,
        loss_log_path=log_dir / "trainer_log.jsonl",
    )

    t0 = time.time()
    with VramMonitor() as vram:
        result = trainer.train()
    elapsed = time.time() - t0

    meta = {
        "run_name": run_name,
        "config_file": str(args.config),
        "smoke": args.smoke,
        "kd_alpha": alpha,
        "kd_temperature": temperature,
        "student_model": cfg["student_model"],
        "teacher_model": cfg["teacher_model"] if alpha > 0 else None,
        "data": ds.stats,
        "student_params_M": round(n_param / 1e6, 1),
        "train_seconds": round(elapsed, 1),
        "train_minutes": round(elapsed / 60, 2),
        "peak_vram_mib": vram.peak_mib,
        "global_steps": result.global_step,
        "final_train_loss": round(result.training_loss, 4),
        "hparams": {k: cfg[k] for k in (
            "num_train_epochs", "per_device_train_batch_size",
            "gradient_accumulation_steps", "learning_rate", "cutoff_len",
            "lr_scheduler_type", "warmup_ratio", "optim") if k in cfg},
    }
    (log_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    if not args.smoke:
        out_dir.mkdir(parents=True, exist_ok=True)
        # ★ 存 fp16 而不是训练时的 fp32：0.5B fp32 是 2GB，fp16 是 1GB，
        #   评测和部署都用 fp16，没必要留一份 fp32 副本占磁盘。
        student.half()
        student.config.use_cache = True
        student.save_pretrained(str(out_dir), safe_serialization=True)
        tok.save_pretrained(str(out_dir))
        print(f"[save] 学生权重(fp16) -> {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
