"""
step3_eval.py — Week8 Day41 / 任务书 41.1
自动加载合并后的模型 → 跑基准评测（CEval/CMMLU）+ 20 题自定义人工评估集
（自动算 5 维分）→ 汇总成一张 CSV。
Loads the merged model, runs the CEval/CMMLU benchmark and the 20-question
custom set (auto-scored on 5 dimensions), and emits one summary CSV.

===========================================================================
★ 最重要的一条：这个脚本**不会**在评测跑不起来时编个分数出来
===========================================================================
    任务书写的是"运行 OpenCompass（CEval/CMMLU）并解析结果"。本机的真实情况是：
    **OpenCompass 从来没有在这台机器上成功跑起来过**——第 3 周 Day15 就卡住了，
    `Week3/deliverables/OpenCompass评测分数表.md` 至今整张表都是 ⏳，
    `Week3/code/run_opencompass.md` 里记着排障过程。所以本脚本的基准评测部分
    实现成**三级回退**，每一级都把"用了哪个后端、为什么"写进 CSV 的
    `bench_backend` / `bench_note` 两列：

      ① opencompass —— 装了就用（`import opencompass` 成功且能定位 CLI）
      ② llamafactory —— LF 自带的 5-shot MCQA 评测器，读同一份 CEval/CMMLU 数据
      ③ unavailable —— 两条路都不通，如实记 `unavailable` + 具体原因

    ★ 绝不允许出现第四种情况："跑不起来，于是填一个看起来合理的数字"。
      一份编造的 CEval 分数比一个空格危险得多：空格会让人去查，数字不会。
      CSV 里的 `⏳/unavailable` 就是它该有的样子。
      The benchmark stage NEVER fabricates a score. If no backend works,
      the CSV records `unavailable` plus the concrete reason.

★ 为什么自定义 20 题这一段是"必跑"的，而基准评测是"可选"的
    20 题集完全自给自足：题目在 `Week3/data/eval_questions.json`，打分规则在
    `Week8/configs/eval.yaml`，两者都在仓库里，不联网、不下数据包。
    CEval/CMMLU 要 ~1.6GB 数据包 + 能用的评测框架，在干净环境里未必立刻具备。
    验收标准 ❶ 要求"干净环境可无报错运行（至少包括数据准备和评估步骤）"——
    所以评估步骤的**主干**必须是那个一定跑得通的部分，基准评测降级为增强项。
    `--skip-bench` 是默认行为的显式化，不是逃避。

★ 关于自动 5 维分的性质（详见 auto_score.py 文件头）
    它是人工评分的**代理指标**，不是替代品。实测对齐质量（`auto_score.py --validate`
    拿第 3 周 100 条真实人工打分做的回归）：
        模型级 Spearman rho = 0.90（5 个模型排序几乎一致，仅第 3/4 名互换）
        题级 rho = 0.699，题级平均绝对偏差 MAE = 0.376
    这个精度足以当**回归护栏**（改版掉分能立刻发现），不足以下"A 比 B 好 0.1 分"
    这种结论。CSV 里因此同时给出 5 个维度分而不只是一个总分——维度分掉在哪一维，
    比总分掉了多少更有诊断价值。

★ 显存礼让
    生成阶段要占约 6.5GB（3B bf16 + KV）。本脚本在加载模型前先查空闲显存，
    不够就直接失败退出并打印当前占用——而不是让 CUDA 在半路 OOM，
    把一次几分钟的生成浪费掉。阈值见 `--min-free-gb`。

用法 / Usage（仓库根目录）:
    # 最常用：评一个模型，只跑 20 题自定义集
    .venv/Scripts/python.exe Week8/scripts/step3_eval.py \
        --model models/Qwen2.5-3B-week8-dpo-merged --tag week8_dpo

    # 一次评多个模型（tag=path），追加进同一张 CSV
    .venv/Scripts/python.exe Week8/scripts/step3_eval.py \
        --models base=models/Qwen2.5-3B-Instruct week8_dpo=models/Qwen2.5-3B-week8-dpo-merged

    # 带上基准评测（会自动探测后端）
    .venv/Scripts/python.exe Week8/scripts/step3_eval.py --model ... --tag ... --bench

    # 不重新生成，直接给已有答卷打分（第 3 周的答卷可以直接喂进来）
    .venv/Scripts/python.exe Week8/scripts/step3_eval.py \
        --reuse-answers Week3/deliverables/eval_answers/answers_qwen_base.json --tag w3_base

    # 冒烟：只跑 2 题，验证调用链
    .venv/Scripts/python.exe Week8/scripts/step3_eval.py --model ... --tag smoke --quick

产物 / Output:
    Week8/deliverables/eval_summary.csv          汇总表（任务书 41.1 的交付物）
    Week8/deliverables/eval_summary.md           同一份数据的人读版
    Week8/deliverables/eval_details/<tag>.json   逐题 5 维分 + 判定明细
    Week8/deliverables/eval_answers/<tag>.json   逐题原始答案（可复现打分）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))   # 为了 import auto_score

QUESTIONS = ROOT / "Week3" / "data" / "eval_questions.json"

# ★ 输出根目录可以用环境变量 WEEK8_DELIV_DIR 覆盖。
#   动机很具体：`verify_all.sh` 的冒烟检查会真的跑一遍打分，如果它写进正式的
#   deliverables，`eval_summary.csv` 里就会混进 verify_smoke / verify_gen_smoke
#   这类假条目——**自检把它要保护的交付物弄脏了**（2026-08-25 实测，
#   一次 --full 自检往 CSV 里塞了 4 行垃圾）。
#   自检负责证明流程能跑，不负责产出成绩单，两者的落盘位置必须分开。
#   A self-check must not mutate the artifacts it is checking.
DELIV = Path(os.environ.get("WEEK8_DELIV_DIR") or (ROOT / "Week8" / "deliverables"))
CSV_PATH = DELIV / "eval_summary.csv"
MD_PATH = DELIV / "eval_summary.md"
DETAIL_DIR = DELIV / "eval_details"
ANSWER_DIR = DELIV / "eval_answers"
OC_RAW = DELIV / "oc_raw"          # 基准评测的原始输出（.gitignore 掉）

# 与 Week3/code/eval_harness.py 完全一致：同一上限、贪心解码。
# 不一致的话，Week8 的分数就没法和第 3 周的答卷横向比了。
MAX_NEW_TOKENS = 512

DIMS = ["accuracy", "completeness", "logic", "safety", "format"]
DIM_CN = {"accuracy": "准确性", "completeness": "完整性", "logic": "逻辑性",
          "safety": "安全性", "format": "格式"}

# CSV 列顺序。定死在这里而不是靠 dict 顺序——追加写入时必须和已有表头对齐。
CSV_FIELDS = [
    "tag", "model_path", "timestamp",
    "bench_backend", "ceval_avg", "ceval_stem", "ceval_social",
    "ceval_humanities", "ceval_other", "cmmlu_avg", "bench_note",
    "accuracy", "completeness", "logic", "safety", "format", "total",
    "n_questions", "n_keyed", "gen_seconds", "tok_per_s",
]


# ===========================================================================
# 小工具
# ===========================================================================
def log(msg: str) -> None:
    print(f"[step3][{datetime.now():%H:%M:%S}] {msg}", flush=True)


def free_vram_gb() -> float | None:
    """返回当前空闲显存 GiB。拿不到（无卡 / 无 nvidia-smi）返回 None。

    ★ 用 nvidia-smi 而不是 torch.cuda.mem_get_info：这一步发生在 import torch
      之前（torch 首次 import + CUDA 初始化要十几秒），失败时早退能省掉这段。
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        used, total = (int(x) for x in r.stdout.strip().splitlines()[0].split(","))
        return (total - used) / 1024.0
    except Exception:
        return None


def load_questions(limit: int = 0) -> list[dict]:
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    qs = data["questions"]
    return qs[:limit] if limit else qs


# ===========================================================================
# 阶段一：基准评测（CEval / CMMLU）—— 三级回退，见文件头
# ===========================================================================
def detect_bench_backend() -> tuple[str, str]:
    """探测可用的基准评测后端，返回 (backend, note)。

    探测顺序 opencompass → llamafactory → ceval_local → unavailable。
    note 里写清楚"为什么是这个"，它会原样进 CSV，读表的人不用再来读代码。

    ★ 第三级 `ceval_local` 是本机实际生效的那一级（2026-08-25 实测）。
      前两级在这台机器上都是死路，而且死因不同：
        · opencompass —— 从第 3 周 Day15 起就没装成功过
        · llamafactory —— 本地版本 0.9.6.dev0 @ 76a0391 上游**已删除
          `evaluation/` 数据加载目录**（`git ls-files | grep ^evaluation` 为空），
          于是 `cached_file('evaluation/ceval', 'mapping.json')` 报
          "not a local folder and is not a valid model identifier"。
      但卡住的其实**不是评测方法，而是数据**——而数据一行 `load_dataset` 就下来了。
      所以第三级是我们自己写的一百来行评测器（见 ceval_local.py），
      它把交付文档里那 52 个 ⏳ 变成了真实分数。
      Tier 3 is what actually works here: the blocker was the data, not the method.
    """
    try:
        import opencompass                                    # noqa: F401
        return "opencompass", "opencompass 已安装"
    except ImportError:
        oc_note = "opencompass 未安装"

    lf_ok = False
    try:
        import llamafactory                                   # noqa: F401
        lf_ok = True
    except ImportError:
        pass
    if lf_ok:
        # 光能 import 不够——LF 新版删了 evaluation/ 目录，数据根本加载不了。
        # 与其起一个必然失败的子进程（要先加载模型，几十秒），不如先探数据。
        try:
            from transformers.utils import cached_file
            cached_file(path_or_repo_id="evaluation/ceval", filename="mapping.json")
            return "llamafactory", f"{oc_note}，回退到 LLaMA-Factory 5-shot MCQA 评测器"
        except Exception:
            lf_note = ("llamafactory 已安装但其 evaluation/ 数据目录不可用"
                       "（本地 LF 版本上游已删除该目录）")
    else:
        lf_note = "llamafactory 未安装"

    try:
        import datasets                                       # noqa: F401
        return "ceval_local", (f"{oc_note}；{lf_note}；"
                               f"改用自带评测器 ceval_local.py（ppl-5shot 口径，"
                               f"官方 ceval/ceval-exam val 划分）")
    except ImportError:
        return "unavailable", f"{oc_note}；{lf_note}；datasets 也不可用，无法取数据"


def run_bench_llamafactory(model_path: Path, tag: str, task: str,
                           n_shot: int, batch_size: int) -> dict:
    """用 LLaMA-Factory 自带评测器跑 ceval / cmmlu。

    ★ 解析对象是 `<save_dir>/results.log`，不是 stdout。
      LF 的 `Evaluator._save_results()`（eval/evaluator.py:139-154）把
      `f"{category_name:>15}: {100*mean:.2f}"` 同时 print 到 stdout 和写进
      results.log。**以文件为准**：stdout 里混着 tqdm 的进度条回车，
      在 Windows 控制台上经常把分数行冲掉半截。
      Parse results.log, not stdout: tqdm carriage returns corrupt the latter.

    ★ save_dir 必须不存在。LF 的 EvaluationArguments.__post_init__ 对已存在的
      save_dir 直接 raise（hparams/evaluation_args.py:58-60）。所以这里带时间戳。
    """
    save_dir = OC_RAW / f"lf_{tag}_{task}_{datetime.now():%Y%m%d_%H%M%S}"
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "llamafactory.cli", "eval",
           f"--model_name_or_path={model_path}",
           "--template=qwen",
           f"--task={task}",
           f"--n_shot={n_shot}",
           f"--batch_size={batch_size}",
           "--lang=zh",
           f"--save_dir={save_dir}"]
    log(f"[bench/{task}] {' '.join(cmd[-6:])}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, cwd=str(ROOT), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")

    log_file = save_dir / "results.log"
    if not log_file.exists():
        tail = (r.stdout or "")[-600:] + (r.stderr or "")[-1200:]
        return {"ok": False,
                "note": f"LF eval 未产出 results.log（exit={r.returncode}）",
                "stderr_tail": tail}

    # results.log 每行形如 "        Average: 65.43"，类别名右对齐到 15 列
    scores: dict[str, float] = {}
    for line in log_file.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*(.+?):\s*([\d.]+)\s*$", line)
        if m:
            scores[m.group(1).strip()] = float(m.group(2))
    if not scores:
        return {"ok": False, "note": f"results.log 存在但解析不出分数：{log_file}"}
    return {"ok": True, "scores": scores, "raw": str(log_file)}


def run_bench_opencompass(model_path: Path, tag: str) -> dict:
    """用 OpenCompass 跑 ceval_gen + cmmlu_gen，解析 summary CSV。

    ★ 诚实标注：这条分支**在本机从未被执行过**——OpenCompass 至今没在这台
      Windows 机器上装成功（见第 3 周 Day15 的排障记录）。它是按 OC 的标准
      产物布局（`<workdir>/summary/summary_<时间戳>.csv`，列为
      dataset,version,metric,mode,<模型名>）写的，一旦找不到 summary
      就**明确失败**并把 workdir 打出来，而不是返回一个空结果假装成功。
      This branch has never run on this machine; it fails loudly rather than
      silently returning empty scores.
    """
    workdir = OC_RAW / f"oc_{tag}_{datetime.now():%Y%m%d_%H%M%S}"
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "opencompass.cli.main",
           "--hf-type", "chat",
           "--hf-path", str(model_path),
           "--datasets", "ceval_gen", "cmmlu_gen",
           "-w", str(workdir)]
    log(f"[bench/opencompass] {' '.join(cmd)}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, cwd=str(ROOT), env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")

    summaries = sorted(workdir.rglob("summary/summary_*.csv"))
    if not summaries:
        tail = (r.stdout or "")[-600:] + (r.stderr or "")[-1200:]
        return {"ok": False,
                "note": f"OpenCompass 未产出 summary CSV（exit={r.returncode}），workdir={workdir}",
                "stderr_tail": tail}

    scores: dict[str, float] = {}
    with summaries[-1].open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = (row.get("dataset") or "").strip()
            # 最后一列是模型名对应的分数；列名不固定，取除已知元数据列之外的第一个
            val = next((v for k, v in row.items()
                        if k not in ("dataset", "version", "metric", "mode") and v), None)
            try:
                scores[ds] = float(val)
            except (TypeError, ValueError):
                continue
    if not scores:
        return {"ok": False, "note": f"summary CSV 解析不出分数：{summaries[-1]}"}
    return {"ok": True, "scores": scores, "raw": str(summaries[-1])}


def bench_to_columns(backend: str, note: str, ceval: dict, cmmlu: dict) -> dict:
    """把两个 task 的原始分数映射成 CSV 的固定列。缺的一律留 '⏳'，不填 0。

    ★ 为什么缺失值写 '⏳' 而不是空字符串或 0
      0 会被下游当成"考了 0 分"参与平均；空字符串在 Excel 里和"真的是 0"
      长得太像。'⏳' 沿用第 3 周分数表里的记号，语义明确：这一格还没有数。
    """
    out = {k: "⏳" for k in ("ceval_avg", "ceval_stem", "ceval_social",
                            "ceval_humanities", "ceval_other", "cmmlu_avg")}
    out["bench_backend"] = backend
    notes = [note]

    if ceval.get("ok"):
        s = ceval["scores"]
        # LF 的类别名 / OC 的数据集名，两套都试
        out["ceval_avg"] = s.get("Average", s.get("ceval", "⏳"))
        out["ceval_stem"] = s.get("STEM", "⏳")
        out["ceval_social"] = s.get("Social Sciences", "⏳")
        out["ceval_humanities"] = s.get("Humanities", "⏳")
        out["ceval_other"] = s.get("Other", "⏳")
    elif ceval:
        notes.append(f"ceval: {ceval.get('note', '失败')}")

    if cmmlu.get("ok"):
        s = cmmlu["scores"]
        out["cmmlu_avg"] = s.get("Average", s.get("cmmlu", "⏳"))
    elif cmmlu:
        notes.append(f"cmmlu: {cmmlu.get('note', '失败')}")

    out["bench_note"] = "；".join(n for n in notes if n)
    return out


# ===========================================================================
# 阶段二：20 题自定义集 —— 生成答案
# ===========================================================================
def generate_answers(model_path: Path, questions: list[dict], min_free_gb: float) -> list[dict]:
    """加载模型跑完 20 题，返回逐题记录（格式与 Week3 的答卷完全一致）。

    ★ 贪心解码（do_sample=False）不是为了分数好看，是为了**可复现**：
      自动打分器是回归护栏，护栏本身必须是确定的。采样解码下同一个模型
      两次跑分能差 0.2，那护栏就废了。
    """
    free = free_vram_gb()
    if free is not None and free < min_free_gb:
        sys.exit(f"[FAIL] 空闲显存 {free:.1f} GiB < 要求的 {min_free_gb} GiB。\n"
                 f"       3B bf16 生成大约要 6.5 GiB。先关掉占卡的进程，"
                 f"或用 --min-free-gb 显式放宽（放宽不会变魔术，只会让它在中途 OOM）。")
    log(f"空闲显存 {('%.1f GiB' % free) if free is not None else '未知（无 nvidia-smi）'}")

    import torch                                   # 延迟 import：早退时省掉十几秒
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"加载模型：{model_path}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.eval()
    log(f"加载完成 {time.time() - t0:.1f}s")

    records = []
    for i, q in enumerate(questions, 1):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": q["question"]}],
            tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        n_new = int(out.shape[1] - inputs["input_ids"].shape[1])
        answer = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
        dt = round(time.time() - t1, 2)
        records.append({"question_id": q["id"], "category": q["category"],
                        "answer": answer, "gen_seconds": dt, "new_tokens": n_new})
        log(f"  {i}/{len(questions)} {q['id']}  {dt}s  {n_new}tok")

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return records


# ===========================================================================
# 汇总输出
# ===========================================================================
def append_csv(row: dict) -> None:
    """把一行结果追加进 CSV。表头只在文件不存在时写一次。

    ★ newline="" 是 csv 模块在 Windows 上的硬性要求：不加的话每行之间会多出
      一个空行（\\r\\r\\n），Excel 打开是隔行的，pandas 读进来多一堆 NaN 行。
    """
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def render_md() -> None:
    """把 CSV 渲染成人读的 Markdown 表。CSV 给机器，MD 给评审。"""
    if not CSV_PATH.exists():
        return
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    lines = [
        "# Week8 评估汇总表 / Evaluation Summary",
        "",
        f"> 由 `Week8/scripts/step3_eval.py` 自动生成 · 最后更新 {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## 一、自定义 20 题集（自动 5 维打分）",
        "",
        "> 打分规则见 `Week8/scripts/auto_score.py` 与 `Week8/configs/eval.yaml`。",
        "> 这是人工评分的**代理指标**：对第 3 周 100 条真实人工打分回归，"
        "模型级 Spearman rho = 0.90，题级 MAE = 0.376。",
        "> 可用于回归护栏（掉分立刻可见），不足以下「A 比 B 好 0.1 分」这种结论。",
        "",
        "| 模型 tag | 准确性 | 完整性 | 逻辑性 | 安全性 | 格式 | **加权总分** | 题数 | 精确档 | 生成 tok/s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['tag']}` | {r['accuracy']} | {r['completeness']} | {r['logic']} | "
            f"{r['safety']} | {r['format']} | **{r['total']}** | {r['n_questions']} | "
            f"{r['n_keyed']} | {r['tok_per_s']} |")

    lines += [
        "",
        "## 二、基准评测（CEval / CMMLU）",
        "",
        "| 模型 tag | 后端 | CEval avg | STEM | 社科 | 人文 | 其他 | CMMLU avg | 说明 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['tag']}` | {r['bench_backend']} | {r['ceval_avg']} | {r['ceval_stem']} | "
            f"{r['ceval_social']} | {r['ceval_humanities']} | {r['ceval_other']} | "
            f"{r['cmmlu_avg']} | {r['bench_note']} |")

    lines += [
        "",
        "> **⏳ 的含义**：这一格没有数，不是 0 分。基准评测后端不可用时本脚本",
        "> 如实留空并在「说明」列写明原因——编一个看起来合理的分数比留空危险得多。",
        "",
        "## 三、逐题明细",
        "",
        "每个 tag 的逐题 5 维分与判定依据（命中了哪条规则、用的哪一档匹配）在",
        "`Week8/deliverables/eval_details/<tag>.json`；原始答卷在",
        "`Week8/deliverables/eval_answers/<tag>.json`，可用",
        "`auto_score.py --answers <file> --detail` 复现任意一题的打分过程。",
        "",
    ]
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[ok] {MD_PATH}")


# ===========================================================================
# 主流程
# ===========================================================================
def eval_one(tag: str, model_path: Path | None, reuse: Path | None,
             args) -> dict:
    from auto_score import AutoScorer

    questions = load_questions(limit=2 if args.quick else 0)

    # ---- 取答卷：复用已有的，或现场生成 ----
    if reuse:
        payload = json.loads(reuse.read_text(encoding="utf-8"))
        records = payload["records"] if isinstance(payload, dict) else payload
        if args.quick:
            records = records[:2]
        log(f"[{tag}] 复用已有答卷 {reuse}（{len(records)} 题）")
    else:
        records = generate_answers(model_path, questions, args.min_free_gb)
        ANSWER_DIR.mkdir(parents=True, exist_ok=True)
        (ANSWER_DIR / f"{tag}.json").write_text(
            json.dumps({"run_id": tag, "model": str(model_path), "records": records},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 打分 ----
    scorer = AutoScorer()
    res = scorer.score_records(records)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    (DETAIL_DIR / f"{tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    gen_s = round(sum(r.get("gen_seconds", 0) for r in records), 1)
    n_tok = sum(r.get("new_tokens", 0) for r in records)
    row = {
        "tag": tag,
        "model_path": str(model_path or reuse),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **{d: res["dims"][d] for d in DIMS},
        "total": res["dims"]["total"],
        "n_questions": res["n_questions"],
        "n_keyed": res["n_keyed"],
        "gen_seconds": gen_s or "⏳",
        "tok_per_s": round(n_tok / gen_s, 1) if gen_s else "⏳",
    }

    # ---- 基准评测 ----
    if args.bench and model_path is not None:
        backend, note = detect_bench_backend()
        log(f"[{tag}] 基准评测后端：{backend}（{note}）")
        ceval = cmmlu = {}
        if backend == "opencompass":
            oc = run_bench_opencompass(model_path, tag)
            if oc.get("ok"):
                ceval = {"ok": True, "scores": {k: v for k, v in oc["scores"].items()
                                                if "ceval" in k.lower()}}
                cmmlu = {"ok": True, "scores": {k: v for k, v in oc["scores"].items()
                                                if "cmmlu" in k.lower()}}
            else:
                ceval = cmmlu = oc
        elif backend == "llamafactory":
            ceval = run_bench_llamafactory(model_path, tag, "ceval",
                                           args.n_shot, args.bench_batch_size)
            cmmlu = run_bench_llamafactory(model_path, tag, "cmmlu",
                                           args.n_shot, args.bench_batch_size)
        elif backend == "ceval_local":
            # ★ 只有 CEval，没有 CMMLU：本脚本刻意不给 CMMLU 编一个近似值。
            #   ceval_local.py 是照 CEval 的题目结构（52 学科 / dev 5-shot 池 /
            #   val 带答案）写的，CMMLU 的组织方式不同，套用会得到一个看起来
            #   像分数、实际口径不明的数。CMMLU 那一格继续留 ⏳。
            from ceval_local import run_ceval
            try:
                r = run_ceval(model_path, tag, n_shot=args.n_shot,
                              limit=args.ceval_limit, verbose=True)
                ceval = {"ok": True, "scores": {
                    "Average": r["Average"], "STEM": r["STEM"],
                    "Social Sciences": r["Social Sciences"],
                    "Humanities": r["Humanities"], "Other": r["Other"]}}
                note += f"；CEval {r['n_questions']} 题 / {r['seconds']}s"
            except Exception as e:                       # 数据下不来、显存不够…
                ceval = {"ok": False, "note": f"ceval_local 失败：{type(e).__name__}: {e}"}
            cmmlu = {"ok": False, "note": "CMMLU 未实现（不编近似值，见代码注释）"}
        row.update(bench_to_columns(backend, note, ceval, cmmlu))
    else:
        reason = "--bench 未开启（默认只跑自定义题集，理由见脚本文件头）" \
            if not args.bench else "复用答卷模式下无法跑基准评测（没有模型路径）"
        row.update(bench_to_columns("skipped", reason, {}, {}))

    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Week8 Day41 自动评估：基准评测 + 20 题自定义集 → CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, help="待评模型目录")
    ap.add_argument("--tag", type=str, help="该模型在 CSV 里的标识")
    ap.add_argument("--models", nargs="*", default=[],
                    help="批量：tag=path tag=path ...")
    ap.add_argument("--reuse-answers", type=Path,
                    help="不生成，直接给已有答卷打分（需配合 --tag）")
    ap.add_argument("--bench", action="store_true",
                    help="额外跑 CEval/CMMLU（自动探测后端；不可用时如实记录）")
    ap.add_argument("--n-shot", type=int, default=5, help="基准评测 few-shot 数")
    ap.add_argument("--bench-batch-size", type=int, default=4)
    ap.add_argument("--ceval-limit", type=int, default=0,
                    help="ceval_local 后端下每个学科最多评几题（0=全部 1346 题）")
    ap.add_argument("--quick", action="store_true", help="冒烟：只跑 2 题")
    ap.add_argument("--min-free-gb", type=float, default=8.0,
                    help="生成前要求的最小空闲显存（GiB）")
    ap.add_argument("--fresh", action="store_true",
                    help="重写 CSV（默认追加）")
    args = ap.parse_args()

    targets: list[tuple[str, Path | None, Path | None]] = []
    if args.reuse_answers:
        if not args.tag:
            sys.exit("[FAIL] --reuse-answers 必须配 --tag")
        targets.append((args.tag, None, args.reuse_answers))
    if args.model:
        if not args.tag:
            sys.exit("[FAIL] --model 必须配 --tag")
        targets.append((args.tag, Path(args.model), None))
    for spec in args.models:
        if "=" not in spec:
            sys.exit(f"[FAIL] --models 的格式是 tag=path，收到：{spec}")
        t, p = spec.split("=", 1)
        targets.append((t, Path(p), None))
    if not targets:
        sys.exit("[FAIL] 至少要给一个 --model/--tag、--models 或 --reuse-answers")

    for _, p, _ in targets:
        if p is not None and not p.exists():
            sys.exit(f"[FAIL] 模型目录不存在：{p}\n"
                     f"       它一般是 step2_train.sh 的合并产物；先跑训练，"
                     f"或用 --model 指向一个已有模型。")

    if args.fresh and CSV_PATH.exists():
        CSV_PATH.unlink()
        log(f"--fresh：已删除旧 CSV")

    for tag, mpath, reuse in targets:
        log(f"=== {tag} ===")
        row = eval_one(tag, mpath, reuse, args)
        append_csv(row)
        log(f"[{tag}] 加权总分 {row['total']}  "
            + "  ".join(f"{DIM_CN[d]}={row[d]}" for d in DIMS))

    render_md()
    log(f"[ok] {CSV_PATH}")


if __name__ == "__main__":
    main()
