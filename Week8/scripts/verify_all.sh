#!/usr/bin/env bash
# ============================================================================
#  verify_all.sh — Week8 自检脚本 / Self-check for the whole Week8 toolchain
#
#  把「这套东西到底能不能跑」变成一条命令。每一项都是**真的执行**，
#  不是检查文件存不存在——文件存在和文件能跑是两回事。
#  Every check actually EXECUTES; presence checks prove nothing.
#
#  用法 / Usage（仓库根目录）:
#    bash Week8/scripts/verify_all.sh            # 快检：不占 GPU，约 1 分钟
#    bash Week8/scripts/verify_all.sh --full     # 全检：加 GPU 冒烟，约 10 分钟
#    bash Week8/scripts/verify_all.sh --list     # 只列出会检查哪些项
#
#  ---------------------------------------------------------------------------
#  ★ 取舍一：为什么分「快检 / 全检」两档
#    快检的定位是**改完代码随手跑一下**——它必须快到你愿意每次都跑。
#    占 GPU 的检查（真加载模型生成）动辄几分钟，放进默认档会让人跳过它，
#    整个自检就形同虚设。所以默认档只验「调用链通不通」，
#    GPU 冒烟放进 --full，交付前跑一次。
#    The default tier must be fast enough that you actually run it.
#
#  ★ 取舍二：为什么不用 set -e
#    自检的目的是**把所有失败项一次列全**，而不是遇到第一个就停。
#    set -e 之下修 5 个问题要跑 5 遍。这里逐项记退出码，最后汇总。
#
#  ★ 取舍三：为什么用 --dry-run 而不是真跑训练/部署
#    step2 真跑要几十分钟并占满卡，step4 会留下常驻服务。
#    它们的 --dry-run 走的是**完全相同的参数解析、配置加载、路径校验**逻辑，
#    只在最后一步不执行——足以证明「命令拼对了、配置读到了、路径存在」。
#    真跑的证据另存于 Week8/deliverables/logs/ 的验收日志。
#
#  ★ 取舍四（本文件最重要的一条）：路径**只能**通过环境变量传给子进程
#    本仓库路径是 C:/Users/Ruibo's Desktop/... —— 那个撇号会把内嵌的单引号串
#    直接截断。第一版把 $REPO 拼进 `python -c "..."` 的代码串里，
#    21 项检查同时报 SyntaxError: unterminated string literal / unexpected EOF。
#    这已经是本项目**第三次**栽在同一个撇号上：
#      ① 主控 run_pipeline.sh 的 $EVAL_ARGS 按空格分词，把 Ruibo's / Desktop 拆开
#      ② step4_deploy.sh 往 WSL 投递命令时的引号嵌套
#      ③ 本文件
#    环境变量是唯一不受任何一层引号规则影响的传递方式，所以下面所有
#    python -c / sh -c 里的路径一律走 $VR_REPO / os.environ["VR_REPO"]。
#    NEVER interpolate paths into `python -c` / `sh -c` source: the repo path
#    contains an apostrophe. Pass them through the environment instead.
# ============================================================================
set -uo pipefail                  # 见 ★取舍二：故意不加 -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
. "$REPO/Week8/configs/pipeline.env"

# ★取舍四：所有子进程通过这两个环境变量拿路径
export VR_REPO="$REPO"
export VR_PY="$PYTHON"

# ★ 自检的落盘产物必须与正式交付物**分开**。
#   step3_eval.py / ceval_local.py 认这个环境变量作为输出根目录。
#   不隔离的话，一次 --full 自检会往 Week8/deliverables/eval_summary.csv 里
#   塞进 verify_smoke / verify_gen_smoke 这类假条目——
#   **自检把它本来要保护的东西弄脏了**（2026-08-25 实测，混进 4 行垃圾）。
#   自检负责证明流程能跑，不负责产出成绩单。
#   Self-check artifacts go to a scratch dir, never into the real deliverables.
export WEEK8_DELIV_DIR="$REPO/Week8/logs/verify_scratch"
mkdir -p "$WEEK8_DELIV_DIR"

FULL=0
LIST=0
for a in "$@"; do
    case "$a" in
        --full) FULL=1 ;;
        --list) LIST=1 ;;
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[FAIL] 未知参数: $a"; exit 1 ;;
    esac
done

PASS=0; FAIL=0; SKIP=0
FAILED_ITEMS=""
LOG="$LOG_DIR/verify_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$LOG_DIR"

hdr() { printf "\n\033[1m── %s ─────────────────────────────\033[0m\n" "$1"; }

# check <名称> <命令...>
check() {
    local name="$1"; shift
    if [ "$LIST" = "1" ]; then printf "  · %s\n" "$name"; return 0; fi
    printf "  %-50s " "$name"
    local out rc
    out="$("$@" 2>&1)"; rc=$?
    if [ $rc -eq 0 ]; then
        printf "\033[32m✓\033[0m\n"; PASS=$((PASS+1))
    else
        printf "\033[31m✗ (exit=%d)\033[0m\n" "$rc"; FAIL=$((FAIL+1))
        FAILED_ITEMS="$FAILED_ITEMS\n  · $name (exit=$rc)"
        echo "$out" | tail -6 | sed 's/^/       /'
    fi
    { echo "=== $name (exit=$rc) ==="; echo "$out"; } >> "$LOG"
}

skip() {
    if [ "$LIST" = "1" ]; then printf "  · %s（--full 才跑）\n" "$1"; return 0; fi
    printf "  %-50s \033[33m—\033[0m %s\n" "$1" "${2:-需要 --full}"; SKIP=$((SKIP+1))
}

echo "Week8 工具链自检   $( [ "$FULL" = 1 ] && echo '[全检]' || echo '[快检]' )"
echo "仓库: $REPO"
[ "$LIST" = "1" ] || echo "日志: $LOG"

# ── 1. 环境 ────────────────────────────────────────────────────────────
hdr "1. 环境"
check "Python 解释器可用" "$PYTHON" -c 'import sys'
check "torch + CUDA 可用" "$PYTHON" -c 'import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)'
check "llamafactory 可导入" "$PYTHON" -c 'import llamafactory'
check "训练栈版本符合 requirements 锁定" "$PYTHON" -c '
import sys, transformers, peft, trl, datasets, accelerate
want = {"transformers": "4.56.2", "peft": "0.18.1", "trl": "0.24.0",
        "datasets": "4.0.0", "accelerate": "1.11.0"}
got = {"transformers": transformers.__version__, "peft": peft.__version__,
       "trl": trl.__version__, "datasets": datasets.__version__,
       "accelerate": accelerate.__version__}
bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
if bad:
    print("版本漂移 (实际, 期望):", bad); sys.exit(1)
print("五个关键包版本全部匹配")'
check "基座模型目录存在" test -d "$BASE_MODEL"

# ── 2. 脚本自身 ────────────────────────────────────────────────────────
hdr "2. 脚本自身（语法检查）"
for f in "$REPO"/Week8/scripts/*.sh; do
    check "bash -n $(basename "$f")" bash -n "$f"
done
for f in "$REPO"/Week8/scripts/*.py; do
    check "py_compile $(basename "$f")" "$PYTHON" -m py_compile "$f"
done

# ── 3. 配置 ────────────────────────────────────────────────────────────
hdr "3. 配置"
check "pipeline.env 可被 sh(dash) 加载（POSIX 兼容）" \
      sh -c '. "$VR_REPO/Week8/configs/pipeline.env" && [ -n "$REPO" ]'
for y in sft_best dpo_best merge_sft merge_dpo eval distill_kd student_sft_baseline; do
    check "YAML 可解析: $y.yaml" env VR_YAML="$y" "$PYTHON" -c '
import os, yaml
p = os.path.join(os.environ["VR_REPO"], "Week8", "configs",
                 os.environ["VR_YAML"] + ".yaml")
yaml.safe_load(open(p, encoding="utf-8"))'
done
# ★ 这一项检查的是**实验设计的有效性**，不是语法：B/C 两组若有第二个自变量，
#   「C − B 就是蒸馏净效果」这句话立刻失效，整章分析作废。
check "B/C 两组配置只差 kd_alpha（对照组有效性）" "$PYTHON" -c '
import os, sys, yaml
d = os.path.join(os.environ["VR_REPO"], "Week8", "configs")
b = yaml.safe_load(open(os.path.join(d, "student_sft_baseline.yaml"), encoding="utf-8"))
c = yaml.safe_load(open(os.path.join(d, "distill_kd.yaml"), encoding="utf-8"))
allowed = {"run_name", "output_dir", "kd_alpha"}
extra = {k for k in set(b) | set(c) if b.get(k) != c.get(k)} - allowed
if extra:
    print("B/C 配置除 kd_alpha 外还有差异，对照组失效:", extra); sys.exit(1)
if b["kd_alpha"] != 0.0 or c["kd_alpha"] <= 0:
    print("kd_alpha 取值不对: B=%s C=%s" % (b["kd_alpha"], c["kd_alpha"])); sys.exit(1)
print("对照组有效：仅 kd_alpha 不同 (B=%s, C=%s)" % (b["kd_alpha"], c["kd_alpha"]))'

# ── 4. 主控与各段 ──────────────────────────────────────────────────────
hdr "4. 主控与各段（--dry-run / --list）"
check "run_pipeline.sh --list" bash "$REPO/Week8/scripts/run_pipeline.sh" --list
check "run_pipeline.sh --dry-run（全链路）" bash "$REPO/Week8/scripts/run_pipeline.sh" --dry-run
check "run_pipeline.sh --skip-train --dry-run" bash "$REPO/Week8/scripts/run_pipeline.sh" --skip-train --dry-run
check "run_pipeline.sh --only eval --dry-run" bash "$REPO/Week8/scripts/run_pipeline.sh" --only eval --dry-run
check "run_pipeline.sh --from eval --dry-run" bash "$REPO/Week8/scripts/run_pipeline.sh" --from eval --dry-run
check "run_pipeline.sh --quick --dry-run" bash "$REPO/Week8/scripts/run_pipeline.sh" --quick --dry-run
check "根目录薄壳 run_pipeline.sh --list" bash "$REPO/run_pipeline.sh" --list
check "step2_train.sh --dry-run" bash "$REPO/Week8/scripts/step2_train.sh" --dry-run
check "step2_train.sh --stage sft --dry-run" bash "$REPO/Week8/scripts/step2_train.sh" --stage sft --dry-run
check "step4_deploy.sh --dry-run" bash "$REPO/Week8/scripts/step4_deploy.sh" --dry-run
check "step4_deploy.sh --status（未起服务也应正常返回）" \
      sh -c 'bash "$VR_REPO/Week8/scripts/step4_deploy.sh" --status; true'

# ★ 反向用例：错误参数必须**失败**。
#   只测正常路径的自检是假自检——一个把所有参数都当合法的脚本同样能全绿。
check "拒绝非法 --stage（反向用例）" \
      sh -c 'bash "$VR_REPO/Week8/scripts/step2_train.sh" --stage nosuch >/dev/null 2>&1 && exit 1 || exit 0'
check "拒绝非法 --only（反向用例）" \
      sh -c 'bash "$VR_REPO/Week8/scripts/run_pipeline.sh" --only nosuch >/dev/null 2>&1 && exit 1 || exit 0'
check "拒绝非法 --variant（反向用例）" \
      sh -c 'bash "$VR_REPO/Week8/scripts/step4_deploy.sh" --variant nosuch >/dev/null 2>&1 && exit 1 || exit 0'

# ── 5. 评分与评测（不占 GPU）────────────────────────────────────────────
hdr "5. 评分与评测（不占 GPU）"
check "auto_score.py --validate（对 100 条人工打分回归）" \
      "$PYTHON" "$REPO/Week8/scripts/auto_score.py" --validate
check "打分器对齐质量未退化（模型级 rho ≥ 0.85）" "$PYTHON" -c '
import os, re, subprocess, sys
r = subprocess.run([os.environ["VR_PY"],
                    os.path.join(os.environ["VR_REPO"], "Week8", "scripts", "auto_score.py"),
                    "--validate"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
m = re.search(r"rho\s*=\s*([\d.]+)", r.stdout or "")
if not m:
    print("拿不到 rho"); sys.exit(1)
rho = float(m.group(1)); print("模型级 rho =", rho)
sys.exit(0 if rho >= 0.85 else 1)'
check "step3_eval.py 复用答卷打分（不生成）" \
      "$PYTHON" "$REPO/Week8/scripts/step3_eval.py" \
      --reuse-answers "$REPO/Week3/deliverables/eval_answers/answers_qwen_base.json" \
      --tag verify_smoke
check "基准评测后端探测能给出结论" "$PYTHON" -c '
import os, sys
sys.path.insert(0, os.path.join(os.environ["VR_REPO"], "Week8", "scripts"))
from step3_eval import detect_bench_backend
b, n = detect_bench_backend()
print(b, "|", n)
sys.exit(0 if b in ("opencompass", "llamafactory", "ceval_local", "unavailable") else 1)'
check "compare_distill.py 预检（四组模型是否齐备）" \
      "$PYTHON" "$REPO/Week8/scripts/compare_distill.py" --skip-ceval --skip-gen

# ── 6. 图表与报告 ──────────────────────────────────────────────────────
hdr "6. 图表与报告生成"
check "make_report_figs.py（第 6 章三张图）" "$PYTHON" "$REPO/Week8/scripts/make_report_figs.py"
check "make_arch_fig.py（第 7 章架构图）" "$PYTHON" "$REPO/Week8/scripts/make_arch_fig.py"
check "make_distill_fig.py（蒸馏曲线）" "$PYTHON" "$REPO/Week8/scripts/make_distill_fig.py"
check "build_tech_report.py（八章拼装）" \
      "$PYTHON" "$REPO/Week8/scripts/build_tech_report.py" --no-docx
check "技术报告中文字符数 ≥ 6000" "$PYTHON" -c '
import os, re, subprocess, sys
r = subprocess.run([os.environ["VR_PY"],
                    os.path.join(os.environ["VR_REPO"], "Week8", "scripts", "build_tech_report.py"),
                    "--no-docx"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
m = re.search("\u4e2d\u6587\u5b57\u7b26\u6570\\s*(\\d+)", r.stdout or "")
if not m:
    print("拿不到字数"); sys.exit(1)
n = int(m.group(1)); print("中文字符数 =", n)
sys.exit(0 if n >= 6000 else 1)'
check "报告引用的图片文件都存在" "$PYTHON" -c '
import os, re, sys
from pathlib import Path
d = Path(os.environ["VR_REPO"], "Week8", "reports")
md = next(d.glob("\u6280\u672f\u62a5\u544a_*.md"))
refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md.read_text(encoding="utf-8"))
missing = [p for p in refs if not (d / p).exists()]
if missing:
    print("缺图:", missing); sys.exit(1)
print("全部", len(refs), "张图片存在")'

# ── 7. 交付物 ──────────────────────────────────────────────────────────
hdr "7. 交付物"
for f in \
    "Week8/deliverables/data_stats.json" \
    "Week8/deliverables/data_stats.md" \
    "Week8/deliverables/eval_summary.csv" \
    "Week8/deliverables/蒸馏效果对比表.md" \
    "Week8/deliverables/logs/acceptance_run_data.log" \
    "Week8/deliverables/logs/acceptance_run_eval.log" \
    "Week8/logs/retry_history.json" \
    "Week8/reports/技术报告_Qwen2.5-3B全链路实践.md" \
    "Week8/reports/技术报告_Qwen2.5-3B全链路实践.docx" \
    "Week8/docs/Pipeline使用说明.md" \
    "Week8/docs/Day40_数据与训练自动化.md" \
    "Week8/docs/Day41_评估与部署自动化.md" \
    "Week8/docs/Day42_知识蒸馏.md" \
    "Week8/docs/脚本速查.md" \
    "Week8/README.md" \
    "README.md" "requirements.txt" "environment.yml" ".gitattributes" ; do
    check "存在: $f" test -f "$REPO/$f"
done
check "retry_history.json 是合法 JSON 数组" "$PYTHON" -c '
import json, os
p = os.path.join(os.environ["VR_REPO"], "Week8", "logs", "retry_history.json")
d = json.load(open(p, encoding="utf-8"))
assert isinstance(d, list) and d, "not a non-empty list"
print(len(d), "条重试记录")'
check "eval_summary.csv 可被解析且非空" "$PYTHON" -c '
import csv, io, os, sys
p = os.path.join(os.environ["VR_REPO"], "Week8", "deliverables", "eval_summary.csv")
rows = list(csv.DictReader(io.open(p, encoding="utf-8-sig", newline="")))
print(len(rows), "行")
sys.exit(0 if rows else 1)'
# ★ 文档里的相对链接必须真的指向存在的文件。死链是文档最常见也最不该有的缺陷：
#   写的时候路径是对的，重构一次就全断了，而没人会去逐个点开验证。
check "README/文档里的相对链接无死链" "$PYTHON" -c '
import os, re, sys
from pathlib import Path
root = Path(os.environ["VR_REPO"])
docs = [root / "README.md", root / "Week8" / "README.md"]
docs += sorted((root / "Week8" / "docs").glob("*.md"))
bad = []
for d in docs:
    if not d.exists():
        continue
    for _text, target in re.findall(r"\[([^\]]+)\]\(([^)#]+)\)",
                                    d.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (d.parent / target).exists():
            bad.append("%s -> %s" % (d.relative_to(root), target))
if bad:
    print("死链:")
    for b in bad:
        print("  ", b)
    sys.exit(1)
print("检查了", len(docs), "份文档，无死链")'

check ".gitignore 未误伤验收日志" \
      sh -c 'cd "$VR_REPO" && ! git check-ignore -q Week8/deliverables/logs/acceptance_run_eval.log'
check "仓库中没有权重文件入库" "$PYTHON" -c '
import os, subprocess, sys
r = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD"],
                   cwd=os.environ["VR_REPO"], capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
bad = [f for f in r.stdout.splitlines()
       if f.lower().endswith((".safetensors", ".bin", ".gguf", ".pt", ".pth"))]
if bad:
    print("权重文件被提交了:", bad[:5]); sys.exit(1)
print("无权重文件入库")'

# ── 8. GPU 冒烟（--full）──────────────────────────────────────────────
hdr "8. GPU 冒烟"
if [ "$FULL" = "1" ]; then
    check "ceval_local.py 冒烟（2 学科 × 2 题）" \
          "$PYTHON" "$REPO/Week8/scripts/ceval_local.py" \
          --model "$BASE_MODEL" --tag verify_smoke --limit 2 \
          --subjects computer_network operating_system
    check "step3_eval.py 真加载模型生成（2 题）" \
          "$PYTHON" "$REPO/Week8/scripts/step3_eval.py" \
          --model "$BASE_MODEL" --tag verify_gen_smoke --quick
    check "run_pipeline.sh --quick --skip-train（端到端）" \
          bash "$REPO/Week8/scripts/run_pipeline.sh" --quick --skip-train
else
    skip "ceval_local.py 冒烟"
    skip "step3_eval.py 真加载模型生成"
    skip "run_pipeline.sh --quick 端到端"
fi

# ── 汇总 ───────────────────────────────────────────────────────────────
[ "$LIST" = "1" ] && exit 0
echo
printf "\033[1m═══ 结果 ═══\033[0m\n"
printf "  通过 \033[32m%d\033[0m   失败 \033[31m%d\033[0m   跳过 \033[33m%d\033[0m\n" "$PASS" "$FAIL" "$SKIP"
if [ "$FAIL" -gt 0 ]; then
    printf "\n\033[31m失败项：\033[0m%b\n" "$FAILED_ITEMS"
    echo "  完整输出见 $LOG"
    exit 1
fi
[ "$FULL" = "1" ] || echo "  （GPU 冒烟已跳过，交付前请跑一次 --full）"
echo "  日志: $LOG"
exit 0
