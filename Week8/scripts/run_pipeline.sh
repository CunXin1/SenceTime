#!/usr/bin/env bash
# ============================================================================
#  run_pipeline.sh — Week8 Day41 / 任务书 41.3
#  主控脚本：把 step1 数据 → step2 训练 → step3 评测 → step4 部署 四段串起来，
#  支持 --skip-train / --skip-eval 等参数实现分段执行。
#  Master orchestrator chaining the four stages, with per-stage skip flags.
#
#  用法 / Usage（仓库根目录，或用根目录的薄壳 `bash run_pipeline.sh`）:
#    bash Week8/scripts/run_pipeline.sh                    # 全链路
#    bash Week8/scripts/run_pipeline.sh --quick            # 冒烟：小步数走完全链路
#    bash Week8/scripts/run_pipeline.sh --skip-train       # 只做数据 + 评估(+部署)
#    bash Week8/scripts/run_pipeline.sh --skip-deploy
#    bash Week8/scripts/run_pipeline.sh --only eval        # 只跑一段
#    bash Week8/scripts/run_pipeline.sh --from eval        # 从这一段开始往后跑
#    bash Week8/scripts/run_pipeline.sh --dry-run          # 只打印各段将执行的命令
#    bash Week8/scripts/run_pipeline.sh --list             # 列出所有段及其状态
#
#  ---------------------------------------------------------------------------
#  ★ 取舍一：为什么是"跳过"而不是"断点续跑"
#    看起来更聪明的设计是自动判断"产物已存在就跳过"。但这条流水线的四段
#    代价差三个数量级：step1 十几秒、step2 几十分钟、step3 几分钟、step4 常驻。
#    自动跳过意味着**改了配置重跑时它会悄悄用旧产物**——而这正是最容易
#    出错也最难发现的一类 bug（"我明明改了 lr，为什么分数一模一样"）。
#    所以这里选择**显式跳过**：跳过哪一段永远是人写在命令行里的决定，
#    脚本只负责忠实执行并在开头把计划打出来让人核对。
#    Explicit skips, never implicit "output exists so skip" — the latter
#    silently reuses stale artifacts after a config change.
#
#  ★ 取舍二：--skip-train 是**默认推荐**的用法，不是退路
#    验收标准 ❶ 写的是"至少包括数据准备和评估步骤"。原因很实在：
#    完整 SFT+DPO 训练要几十分钟到几小时并且占满一张 24GB 卡，
#    任何人拿到这个仓库都不可能把它当成"验证安装是否正确"的第一条命令。
#    所以 README 的快速开始给的是 `--quick`，验收给的是 `--skip-train`，
#    全链路留给真正要复现训练的人。
#
#  ★ 取舍三：为什么部署段默认**不**跑
#    step4 起的是常驻服务，它会一直占着显存和端口直到被显式停止。
#    一个"跑完会退出"的流水线突然变成"跑完还挂着两个后台进程"，
#    对无人值守调用（CI、夜间批处理）是灾难。所以 deploy 段必须
#    `--with-deploy` 显式打开，且脚本结束时会明确打印怎么停。
#    The deploy stage leaves long-running services behind, so it is opt-in.
#
#  ★ 取舍四：每段的退出码都要检查，且失败即停
#    数据没准备好还去训练、模型没合并出来还去评测——这类"带着错误往前冲"
#    产生的报错信息会指向完全无关的地方（比如报"模型目录不存在"，
#    而真正的原因是三步之前的数据清洗挂了）。这里一段失败就停，
#    并明确打出"是哪一段、日志在哪、下一步该看什么"。
#
#  ★ 取舍五：整条流水线的总日志用 tee 而不是重定向
#    无人值守时要能事后翻日志，交互运行时要能实时看到进度。tee 两者都满足。
#    配合 pipefail，管道里训练脚本的非零退出码不会被 tee 的 0 吃掉。
# ============================================================================
set -uo pipefail                 # 不加 -e：失败时要自己打印诊断再退出

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$REPO/Week8/configs/pipeline.env"

SCRIPTS="$REPO/Week8/scripts"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
PIPE_LOG="$LOG_DIR/pipeline_$RUN_ID.log"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
SKIP_DATA=0; SKIP_TRAIN=0; SKIP_EVAL=0
WITH_DEPLOY=0                     # ★取舍三：部署段默认关
QUICK=0; DRY_RUN=0; LIST=0
ONLY=""; FROM=""
TRAIN_STAGE="all"                 # 透传给 step2_train.sh
EVAL_BENCH=0                      # 透传给 step3_eval.py 的 --bench
DEPLOY_VARIANT="awq"

STAGES="data train eval deploy"

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

参数 / Flags:
  --skip-data          跳过 step1 数据准备
  --skip-train         跳过 step2 训练（验收推荐；理由见脚本 ★取舍二）
  --skip-eval          跳过 step3 评测
  --with-deploy        额外跑 step4 部署（默认关；理由见 ★取舍三）
  --only   <stage>     只跑一段：data / train / eval / deploy
  --from   <stage>     从某段开始往后跑
  --quick              冒烟模式：训练只跑 2 步、评测只跑 2 题，验证调用链
  --train-stage <s>    透传 step2 的 --stage：sft / dpo / all（默认 all）
  --bench              评测时额外跑 CEval/CMMLU（自动探测后端，不可用如实记录）
  --variant <v>        部署哪个后端：fp16 / awq / gptq / vl（默认 awq）
  --dry-run            只打印各段将要执行的命令
  --list               列出所有段与当前产物状态，然后退出
  -h, --help           这段帮助
EOF
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-data)    SKIP_DATA=1; shift ;;
        --skip-train)   SKIP_TRAIN=1; shift ;;
        --skip-eval)    SKIP_EVAL=1; shift ;;
        --with-deploy)  WITH_DEPLOY=1; shift ;;
        --skip-deploy)  WITH_DEPLOY=0; shift ;;   # 默认就是关的，留着是为了写法对称
        --only)         ONLY="${2:-}"; shift 2 ;;
        --only=*)       ONLY="${1#*=}"; shift ;;
        --from)         FROM="${2:-}"; shift 2 ;;
        --from=*)       FROM="${1#*=}"; shift ;;
        --quick)        QUICK=1; shift ;;
        --train-stage)  TRAIN_STAGE="${2:-}"; shift 2 ;;
        --train-stage=*) TRAIN_STAGE="${1#*=}"; shift ;;
        --bench)        EVAL_BENCH=1; shift ;;
        --variant)      DEPLOY_VARIANT="${2:-}"; shift 2 ;;
        --variant=*)    DEPLOY_VARIANT="${1#*=}"; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --list)         LIST=1; shift ;;
        -h|--help)      usage 0 ;;
        *) echo "[FAIL] 未知参数: $1"; echo; usage 1 ;;
    esac
done

valid_stage() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if [ -n "$ONLY" ]; then
    valid_stage "$ONLY" || { echo "[FAIL] --only 只能是: $STAGES（收到 $ONLY）"; exit 1; }
    SKIP_DATA=1; SKIP_TRAIN=1; SKIP_EVAL=1; WITH_DEPLOY=0
    case "$ONLY" in
        data) SKIP_DATA=0 ;; train) SKIP_TRAIN=0 ;;
        eval) SKIP_EVAL=0 ;; deploy) WITH_DEPLOY=1 ;;
    esac
fi

if [ -n "$FROM" ]; then
    valid_stage "$FROM" || { echo "[FAIL] --from 只能是: $STAGES（收到 $FROM）"; exit 1; }
    seen=0
    for s in $STAGES; do
        [ "$s" = "$FROM" ] && seen=1
        if [ "$seen" = "0" ]; then
            case "$s" in
                data) SKIP_DATA=1 ;; train) SKIP_TRAIN=1 ;; eval) SKIP_EVAL=1 ;;
            esac
        elif [ "$s" = "deploy" ]; then
            WITH_DEPLOY=1
        fi
    done
fi

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
log()  { echo "[pipeline][$(date +%H:%M:%S)] $*"; }
rule() { echo "════════════════════════════════════════════════════════════════"; }

# 跑一段。$1=段名 $2...=命令。失败即停（★取舍四）。
STAGE_RESULTS=""
run_stage() {
    local name="$1"; shift
    rule
    log "▶ 段 [$name] 开始"
    log "  $*"
    if [ "$DRY_RUN" = "1" ]; then
        log "  (dry-run) 未执行"
        STAGE_RESULTS="$STAGE_RESULTS $name:dry"
        return 0
    fi
    local t0 rc
    t0="$(date +%s)"
    "$@"
    rc=$?
    local dt=$(( $(date +%s) - t0 ))
    if [ "$rc" -ne 0 ]; then
        log "✗ 段 [$name] 失败（exit=$rc，耗时 ${dt}s）"
        STAGE_RESULTS="$STAGE_RESULTS $name:FAIL"
        rule
        log "流水线在 [$name] 中断。排查顺序："
        log "  1) 上面这一段自己的报错（往上翻，别只看最后一行）"
        log "  2) 该段的日志：$LOG_DIR/"
        log "  3) 全流程日志：$PIPE_LOG"
        log "  4) README.md 的「常见问题」——前 7 周踩过的坑基本都在那"
        log "修好之后可以从这一段续跑： --from $name"
        exit "$rc"
    fi
    log "✓ 段 [$name] 完成（${dt}s）"
    STAGE_RESULTS="$STAGE_RESULTS $name:ok(${dt}s)"
}

# ---------------------------------------------------------------------------
# --list：列出各段与产物状态
# ---------------------------------------------------------------------------
mark() { [ -e "$1" ] && echo "✅" || echo "—"; }
if [ "$LIST" = "1" ]; then
    echo "段         脚本                        关键产物                              状态"
    echo "---------------------------------------------------------------------------------"
    printf "%-10s %-27s %-37s %s\n" "data"   "step1_data_prep.py"  "Week8/data/dataset_info.json"        "$(mark "$DATA_DIR/dataset_info.json")"
    printf "%-10s %-27s %-37s %s\n" ""       ""                    "Week8/deliverables/data_stats.json"  "$(mark "$DELIV_DIR/data_stats.json")"
    printf "%-10s %-27s %-37s %s\n" "train"  "step2_train.sh"      "models/...week8-sft-merged"          "$(mark "$MERGED_SFT")"
    printf "%-10s %-27s %-37s %s\n" ""       ""                    "models/...week8-dpo-merged"          "$(mark "$MERGED_DPO")"
    printf "%-10s %-27s %-37s %s\n" "eval"   "step3_eval.py"       "Week8/deliverables/eval_summary.csv" "$(mark "$DELIV_DIR/eval_summary.csv")"
    printf "%-10s %-27s %-37s %s\n" "deploy" "step4_deploy.sh"     "（常驻服务，无落盘产物）"              "—"
    echo
    echo "各段健康状态可单独查：bash Week8/scripts/step4_deploy.sh --status"
    exit 0
fi

# ---------------------------------------------------------------------------
# 开场：把计划打出来（★取舍一：跳过是人的决定，先让人核对）
# ---------------------------------------------------------------------------
{
rule
log "Week8 全链路 Pipeline  run_id=$RUN_ID"
log "仓库 : $REPO"
log "解释器: $PYTHON"
log "日志 : $PIPE_LOG"
rule
log "本次计划："
log "  step1 数据准备   $([ "$SKIP_DATA" = 1 ]  && echo '跳过' || echo '执行')"
log "  step2 训练       $([ "$SKIP_TRAIN" = 1 ] && echo '跳过' || echo "执行（stage=$TRAIN_STAGE）")"
log "  step3 评测       $([ "$SKIP_EVAL" = 1 ]  && echo '跳过' || echo "执行（bench=$([ "$EVAL_BENCH" = 1 ] && echo on || echo off)）")"
log "  step4 部署       $([ "$WITH_DEPLOY" = 1 ] && echo "执行（variant=$DEPLOY_VARIANT）" || echo '跳过（需 --with-deploy）')"
[ "$QUICK" = "1" ]   && log "  模式：--quick 冒烟（训练 $QUICK_STEPS 步 / 评测 2 题，产物不可用于交付）"
[ "$DRY_RUN" = "1" ] && log "  模式：--dry-run（只打印命令）"
rule

# =========================== step1 数据准备 ===========================
if [ "$SKIP_DATA" = "0" ]; then
    run_stage data "$PYTHON" "$SCRIPTS/step1_data_prep.py"
else
    log "⏭ 段 [data] 按参数跳过"
    STAGE_RESULTS="$STAGE_RESULTS data:skip"
fi

# =========================== step2 训练 ===============================
if [ "$SKIP_TRAIN" = "0" ]; then
    # 同样用数组，理由见下面 eval 段的 ★ 注释
    TRAIN_ARGS=(--stage "$TRAIN_STAGE")
    [ "$QUICK" = "1" ]   && TRAIN_ARGS+=(--quick)
    [ "$DRY_RUN" = "1" ] && TRAIN_ARGS+=(--dry-run)
    run_stage train bash "$SCRIPTS/step2_train.sh" "${TRAIN_ARGS[@]}"
else
    log "⏭ 段 [train] 按参数跳过"
    STAGE_RESULTS="$STAGE_RESULTS train:skip"
fi

# =========================== step3 评测 ===============================
if [ "$SKIP_EVAL" = "0" ]; then
    # ★ 评哪个模型：优先 DPO 合并产物 → 退回 SFT 合并产物 → 再退回基座。
    #   最后这一档很重要：`--skip-train` 时训练产物根本不存在，
    #   如果这里直接失败，验收标准 ❶ 要求的"数据+评估两段能跑通"就实现不了。
    #   退回基座评测同样是有意义的——它就是第 3~4 周的对照基线。
    EVAL_TAG="week8_dpo"; EVAL_MODEL="$MERGED_DPO"
    if [ ! -d "$EVAL_MODEL" ]; then
        EVAL_TAG="week8_sft"; EVAL_MODEL="$MERGED_SFT"
    fi
    if [ ! -d "$EVAL_MODEL" ]; then
        EVAL_TAG="base"; EVAL_MODEL="$BASE_MODEL"
        log "⚠ 训练产物不存在，改评基座 $BASE_MODEL（这是第 3/4 周的对照基线，不是错误）"
    fi
    [ "$QUICK" = "1" ] && EVAL_TAG="${EVAL_TAG}_quick"

    # ★ 参数必须攒进**数组**，不能攒进字符串（2026-08-25 实测踩坑）
    #   第一版写的是 EVAL_ARGS="--model $EVAL_MODEL --tag $EVAL_TAG"，展开时
    #   `$EVAL_ARGS` 按空格分词，而仓库路径是 "C:/Users/Ruibo's Desktop/..."——
    #   "Ruibo's" 和 "Desktop/..." 被拆成两个参数，argparse 报
    #   `unrecognized arguments: Desktop/SenceTime_Weeks1-5/models/...`。
    #   报错信息指向 step3_eval.py，真正的错误却在这里，非常难找。
    #   数组的每个元素是一个 argv，加引号展开 "${arr[@]}" 后不再二次分词。
    #   Accumulate flags in an ARRAY: the repo path contains a space, and
    #   string-splitting would tear it into two argv entries.
    EVAL_ARGS=(--model "$EVAL_MODEL" --tag "$EVAL_TAG")
    [ "$QUICK" = "1" ]      && EVAL_ARGS+=(--quick)
    [ "$EVAL_BENCH" = "1" ] && EVAL_ARGS+=(--bench)
    if [ "$DRY_RUN" = "1" ]; then
        log "  (dry-run) $PYTHON $SCRIPTS/step3_eval.py ${EVAL_ARGS[*]}"
        STAGE_RESULTS="$STAGE_RESULTS eval:dry"
    else
        run_stage eval "$PYTHON" "$SCRIPTS/step3_eval.py" "${EVAL_ARGS[@]}"
    fi
else
    log "⏭ 段 [eval] 按参数跳过"
    STAGE_RESULTS="$STAGE_RESULTS eval:skip"
fi

# =========================== step4 部署 ===============================
if [ "$WITH_DEPLOY" = "1" ]; then
    DEPLOY_ARGS=(--variant "$DEPLOY_VARIANT")
    [ "$DRY_RUN" = "1" ] && DEPLOY_ARGS+=(--dry-run)
    run_stage deploy bash "$SCRIPTS/step4_deploy.sh" "${DEPLOY_ARGS[@]}"
else
    log "⏭ 段 [deploy] 跳过（默认关，需 --with-deploy；理由见脚本 ★取舍三）"
    STAGE_RESULTS="$STAGE_RESULTS deploy:skip"
fi

# =========================== 收尾 =====================================
rule
log "流水线结束。各段结果："
for r in $STAGE_RESULTS; do log "  $r"; done
rule
[ -f "$DELIV_DIR/data_stats.md" ]    && log "数据统计报告 : $DELIV_DIR/data_stats.md"
[ -f "$DELIV_DIR/eval_summary.csv" ] && log "评估汇总表   : $DELIV_DIR/eval_summary.csv"
[ -f "$DELIV_DIR/eval_summary.md" ]  && log "评估汇总(人读): $DELIV_DIR/eval_summary.md"
if [ "$WITH_DEPLOY" = "1" ] && [ "$DRY_RUN" = "0" ]; then
    log "⚠ 服务仍在后台运行。停止： bash Week8/scripts/step4_deploy.sh --variant $DEPLOY_VARIANT --stop"
fi
log "全流程日志: $PIPE_LOG"
} 2>&1 | tee "$PIPE_LOG"

# ★ 管道的退出码：加了 pipefail，所以 tee 前面那个大括号块的非零码会传出来。
exit "${PIPESTATUS[0]}"
