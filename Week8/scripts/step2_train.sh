#!/usr/bin/env bash
# ============================================================================
#  step2_train.sh — Week8 Day40 / 任务书 40.2 + 40.3
#  按第 3/4 周得出的最优超参，无人值守地依次跑完
#      SFT → 合并 → DPO → 合并
#  日志自动落 Week8/logs/<stage>_<timestamp>.log 并同时 tee 到终端；
#  训练因显存不足挂掉时按预设阶梯自动降配重试（40.3）。
#  Runs the full SFT -> merge -> DPO -> merge chain unattended, with an
#  OOM-triggered auto-downscale retry ladder.
#
#  用法 / Usage:
#    bash Week8/scripts/step2_train.sh                    # 全链路
#    bash Week8/scripts/step2_train.sh --stage sft        # 只跑 SFT(+合并)
#    bash Week8/scripts/step2_train.sh --stage dpo
#    bash Week8/scripts/step2_train.sh --dry-run          # 只打印命令，不执行
#    bash Week8/scripts/step2_train.sh --quick            # 小步数冒烟（max_steps=2）
#    bash Week8/scripts/step2_train.sh --max-retries 5
#
#  ---------------------------------------------------------------------------
#  ★ 取舍一：OOM 重试时**等效 batch 必须保持不变**
#    等效 batch = per_device_train_batch_size × gradient_accumulation_steps。
#    第 3 周的冠军超参是在"等效 batch=16"下调出来的，第 4 周是 8。如果重试时
#    只把 per_device batch 减半、不补 accumulation，等效 batch 就从 16 掉到 8：
#    每步看到的样本少一半 → 梯度噪声变大 → 有效学习率相当于被改了 → 训出来的
#    模型和"第 3 周的最优实验"根本不是一回事，而流水线还会若无其事地把它
#    当成最优模型交付。所以本脚本的降配阶梯里，batch 减半和 accum 加倍
#    **永远成对出现**，乘积恒等于配置文件里的原值，并在日志里打出来核对。
#    Every rung halves the micro-batch and doubles accumulation TOGETHER, so
#    the effective batch never changes — otherwise the retried run is a
#    different experiment than the tuned one.
#
#  ★ 取舍二：降 cutoff_len 是最后一档，且会被显式标成「已偏离原实验」
#    降 cutoff 是真的改了实验：它既减少每步的 token 数（等效 batch 在"序列"口径
#    上不变、在"token"口径上减半），又会把长样本截得更短、改变数据分布。
#    但它确实是 batch 已经降到 1 之后唯一还剩的手段，所以放在阶梯末端，
#    触发时日志里打 ★警告，retry_history.json 里 deviates_from_baseline=true。
#    宁可交付一个"知道自己不标准"的模型，也不要交付一个"以为自己标准"的。
#
#  ★ 取舍三：为什么不用 set -e
#    这个脚本的核心逻辑就是"命令失败之后继续做事"（判断是不是 OOM、降配、重跑）。
#    set -e 会在第一次训练失败时直接把脚本杀掉，整套重试机制永远不会被执行到。
#    这里用 set -uo pipefail + 逐处显式检查返回码。
#    pipefail 是必须的：训练命令要 `| tee` 到日志，没有 pipefail 的话
#    $? 拿到的是 tee 的返回码（几乎永远是 0），训练崩了也会被当成成功。
#
#  ★ 取舍四：覆盖参数必须写成 `key=value`，不能写成 `--key value`
#    LLaMA-Factory 的 hparams/parser.py:90-93：当 argv[1] 是 .yaml 时，
#    剩余参数走 `OmegaConf.from_cli(sys.argv[2:])` —— 那是 OmegaConf 的
#    **dotlist 语法**，只认 `key=value`。写成 `--per_device_train_batch_size 2`
#    会被 OmegaConf 当成一个叫 "--per_device_train_batch_size" 的键而报错。
#    这跟直接用 llamafactory-cli 传参的写法不一样，很容易踩。
# ============================================================================

# 见 ★取舍三：故意不加 -e
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$REPO/Week8/configs/pipeline.env"

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
STAGE="all"
DRY_RUN=0
QUICK=0

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --stage)        STAGE="${2:-}"; shift 2 ;;
        --stage=*)      STAGE="${1#*=}"; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --quick)        QUICK=1; shift ;;
        --max-retries)  MAX_RETRIES="${2:-}"; shift 2 ;;
        --max-retries=*) MAX_RETRIES="${1#*=}"; shift ;;
        -h|--help)      usage 0 ;;
        *) echo "[FAIL] 未知参数: $1"; usage 1 ;;
    esac
done

case "$STAGE" in
    sft|dpo|all) ;;
    *) echo "[FAIL] --stage 只能是 sft / dpo / all，收到: $STAGE"; exit 1 ;;
esac

mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
HISTORY_JSONL="$LOG_DIR/retry_history.jsonl"   # 追加型明细，永不重写
# retry_history.json（任务书要求的那份）由 JSONL 每次重新渲染成合法数组

# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
log() { echo "[step2][$(date +%H:%M:%S)] $*"; }

# 从扁平 YAML 里取一个顶层标量。合并配置和训练配置都是我们自己写的、
# 无嵌套无锚点的扁平结构，sed 完全够用，不必为读三个数去装 yq。
yaml_get() {
    sed -n "s/^[[:space:]]*$2:[[:space:]]*\([^#]*\).*/\1/p" "$1" \
        | head -1 | sed 's/[[:space:]]*$//' | tr -d '"'\'''
}

# JSON 字符串转义：路径里可能有反斜杠和引号（Windows），不转义会写出坏 JSON。
json_escape() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

# 把 JSONL 明细渲染成合法的 JSON 数组，供人和 step3 读。
# 用 sed 拼而不是调 python：注入式测试会把 $PYTHON 换成假命令，
# 历史文件的写入不能依赖那个被替换掉的解释器。
rebuild_history_json() {
    [ -f "$HISTORY_JSONL" ] || return 0
    {
        echo "["
        sed -e 's/$/,/' -e '$ s/,$//' "$HISTORY_JSONL"
        echo "]"
    } > "$RETRY_HISTORY"
}

record_attempt() {
    # $1 stage  $2 attempt  $3 rung_desc  $4 overrides  $5 exit_code
    # $6 oom(true/false)  $7 deviates(true/false)  $8 logfile
    printf '  {"run_id":"%s","stage":"%s","attempt":%s,"rung":"%s","overrides":"%s",' \
        "$RUN_ID" "$1" "$2" "$(json_escape "$3")" "$(json_escape "$4")" >> "$HISTORY_JSONL"
    printf '"exit_code":%s,"oom_detected":%s,"deviates_from_baseline":%s,' \
        "$5" "$6" "$7" >> "$HISTORY_JSONL"
    printf '"log":"%s","timestamp":"%s"}\n' \
        "$(json_escape "$8")" "$(date -Iseconds)" >> "$HISTORY_JSONL"
    rebuild_history_json
}

# OOM 判定：退出码非 0 **且** 日志里出现显存不足的特征串。
# 只看退出码不行 —— 语法错、数据集没注册、路径写错也都是非 0，那些重试一万次
# 也不会好，降配重试只会把真正的错误信息埋在三层重试日志底下。
is_oom_log() {
    grep -qiE "CUDA out of memory|OutOfMemoryError|CUDA error: out of memory|HIP out of memory" "$1"
}

# ---------------------------------------------------------------------------
# 降配阶梯（40.3）
# 按「对原实验的扰动从小到大」排序，逐档往下走。不适用的档（比如 batch 已经是 1
# 还要求减半）会被自动跳过，不浪费一次重试名额。
#
#   档 1  eval batch 降到 1   —— 对训练数学**零影响**，只省评估时的峰值显存
#   档 2  train batch 减半 + accum 加倍 —— 等效 batch 不变（★取舍一）
#   档 3  同上再来一次
#   档 4  cutoff_len 减半    —— ★ 偏离原实验，日志会警告（★取舍二）
#   档 5  cutoff_len 再减半  —— ★ 偏离原实验
#
# 产出两个全局数组：RUNG_DESC[] / RUNG_OV[]（OV = override 字符串，空格分隔）
# ---------------------------------------------------------------------------
build_ladder() {
    local cfg="$1"
    local tbs ga ebs cut eff
    tbs="$(yaml_get "$cfg" per_device_train_batch_size)"
    ga="$(yaml_get "$cfg" gradient_accumulation_steps)"
    ebs="$(yaml_get "$cfg" per_device_eval_batch_size)"
    cut="$(yaml_get "$cfg" cutoff_len)"
    : "${tbs:=1}"; : "${ga:=1}"; : "${ebs:=1}"; : "${cut:=2048}"
    eff=$(( tbs * ga ))

    RUNG_DESC=(); RUNG_OV=(); RUNG_DEV=()
    BASE_EFF="$eff"
    log "基线：train_bs=$tbs  accum=$ga  eval_bs=$ebs  cutoff=$cut  → 等效 batch=$eff"

    if [ "$ebs" -gt 1 ]; then
        RUNG_DESC+=("eval batch $ebs→1（不影响训练数学）")
        RUNG_OV+=("per_device_eval_batch_size=1")
        RUNG_DEV+=("false")
    fi

    local i cur_tbs=$tbs cur_ga=$ga
    for i in 1 2; do
        if [ "$cur_tbs" -ge 2 ]; then
            cur_tbs=$(( cur_tbs / 2 )); cur_ga=$(( cur_ga * 2 ))
            RUNG_DESC+=("train_bs→$cur_tbs, accum→$cur_ga（等效 batch 仍 = $(( cur_tbs * cur_ga ))）")
            RUNG_OV+=("per_device_train_batch_size=$cur_tbs gradient_accumulation_steps=$cur_ga per_device_eval_batch_size=1")
            RUNG_DEV+=("false")
        fi
    done

    local cur_cut=$cut
    for i in 1 2; do
        if [ "$cur_cut" -ge 512 ]; then
            cur_cut=$(( cur_cut / 2 ))
            RUNG_DESC+=("★ cutoff_len→$cur_cut（已偏离原实验：每步 token 数减半、长样本被多截）")
            RUNG_OV+=("per_device_train_batch_size=$cur_tbs gradient_accumulation_steps=$cur_ga per_device_eval_batch_size=1 cutoff_len=$cur_cut")
            RUNG_DEV+=("true")
        fi
    done
}

# ---------------------------------------------------------------------------
# 跑一个训练 stage，带 OOM 重试
#   $1 = stage 名（sft / dpo）   $2 = 配置文件路径
# ---------------------------------------------------------------------------
run_train_stage() {
    local stage="$1" cfg="$2"
    if [ ! -f "$cfg" ]; then
        log "[FAIL] 找不到配置 $cfg"
        return 1
    fi

    build_ladder "$cfg"

    # dry-run 时把整条降配阶梯打出来 —— 这是这个脚本最需要被人 review 的部分，
    # 而它只在真的 OOM 时才会执行到，平时根本看不见。
    if [ "$DRY_RUN" -eq 1 ]; then
        log "[$stage] 降配阶梯共 ${#RUNG_OV[@]} 档（最多用前 $MAX_RETRIES 档）："
        local k
        for k in "${!RUNG_OV[@]}"; do
            log "  档$(( k + 1 )): ${RUNG_DESC[$k]}"
            log "         覆盖参数: ${RUNG_OV[$k]}"
        done
    fi

    # --quick：小步数冒烟，只验证"调用链通不通"，产物写到 *_quick 目录，
    # 绝不污染正式的 output_dir（否则一次冒烟会把真训练的 checkpoint 冲掉）。
    local quick_ov=""
    if [ "$QUICK" -eq 1 ]; then
        # eval_strategy=no：跳过评估本身，但 eval_dataset 仍会被加载和 tokenize，
        # 所以「val_size / eval_dataset 是否配错」这类问题照样会在冒烟里暴露
        # （LF 在参数解析阶段就会 raise，根本走不到训练）。
        quick_ov="max_steps=$QUICK_STEPS save_steps=100000 eval_strategy=no overwrite_output_dir=true output_dir=saves/week8/qwen/${stage}_quick"
        log "--quick 冒烟：$quick_ov"
    fi

    local attempt=0 max=$MAX_RETRIES
    while : ; do
        local ov="" desc="baseline（配置文件原值）" dev="false"
        if [ "$attempt" -gt 0 ]; then
            local idx=$(( attempt - 1 ))
            if [ "$idx" -ge "${#RUNG_OV[@]}" ]; then
                log "[FAIL] $stage：降配阶梯已用尽（共 ${#RUNG_OV[@]} 档），不再重试。"
                log "       阶梯走到底还 OOM，基本不是 batch 能救的问题 —— 先看看"
                log "       nvidia-smi 是不是有别的进程占着卡。"
                return 1
            fi
            ov="${RUNG_OV[$idx]}"; desc="${RUNG_DESC[$idx]}"; dev="${RUNG_DEV[$idx]}"
            log "第 $attempt 次重试，降配档位：$desc"
            [ "$dev" = "true" ] && log "★ 警告：本档已偏离第 3/4 周的最优超参，产出的模型不能直接与那两周的结果对比。"
        fi

        local logfile="$LOG_DIR/${stage}_${RUN_ID}.log"
        [ "$attempt" -gt 0 ] && logfile="$LOG_DIR/${stage}_${RUN_ID}_retry${attempt}.log"

        # ★ 覆盖参数是 OmegaConf dotlist（key=value），见 ★取舍四。
        # ★ 必须 python -m llamafactory.cli，不能用 llamafactory-cli.exe（Windows 段错误）。
        local -a cmd=("$PYTHON" -m llamafactory.cli train "$cfg")
        local w
        for w in $quick_ov $ov; do cmd+=("$w"); done

        log "[$stage] attempt=$attempt  日志=$logfile"
        printf '[step2] 命令: '; printf '%q ' "${cmd[@]}"; printf '\n'

        if [ "$DRY_RUN" -eq 1 ]; then
            log "[dry-run] 不执行。"
            return 0
        fi

        # cwd 固定仓库根：配置里的 dataset_dir / output_dir 都是仓库相对路径。
        ( cd "$REPO" && "${cmd[@]}" ) 2>&1 | tee "$logfile"
        local rc=${PIPESTATUS[0]}      # ← 训练进程的退出码，不是 tee 的

        if [ "$rc" -eq 0 ]; then
            log "[$stage] 训练成功（attempt=$attempt）"
            record_attempt "$stage" "$attempt" "$desc" "$quick_ov $ov" "$rc" "false" "$dev" "$logfile"
            return 0
        fi

        local oom="false"
        if is_oom_log "$logfile"; then oom="true"; fi
        record_attempt "$stage" "$attempt" "$desc" "$quick_ov $ov" "$rc" "$oom" "$dev" "$logfile"

        if [ "$oom" != "true" ]; then
            log "[FAIL] $stage 退出码 $rc，但日志里没有显存不足的特征串 —— 这不是 OOM，不重试。"
            log "       降配重试只会把真正的错误埋进更多层日志。请看 $logfile 的尾部。"
            tail -n 20 "$logfile" | sed 's/^/       | /'
            return "$rc"
        fi

        attempt=$(( attempt + 1 ))
        if [ "$attempt" -gt "$max" ]; then
            log "[FAIL] $stage：已重试 $max 次仍 OOM，放弃。历史见 $RETRY_HISTORY"
            return 1
        fi
        log "检测到 CUDA OOM（退出码 $rc）→ 降一档重试（$attempt/$max）"
        # 给驱动一点时间把上一个进程的显存真正还回去；Windows 上进程退出后
        # 显存释放不是瞬时的，立刻重启很容易又撞上同一个 OOM。
        sleep 10
    done
}

# ---------------------------------------------------------------------------
# 合并 stage
# ---------------------------------------------------------------------------
run_merge_stage() {
    local name="$1" cfg="$2"
    local logfile="$LOG_DIR/merge_${name}_${RUN_ID}.log"
    local -a cmd=("$PYTHON" "$REPO/Week8/scripts/merge_model.py" --config "$cfg" --force)

    # 冒烟模式不真合并：一个 3B 模型落盘约 6GB，C 盘余量紧张，而且合并跑不跑
    # 跟"LF 调用链通不通"无关。只让 merge_model.py 走它自己的 --dry-run 打印命令。
    if [ "$QUICK" -eq 1 ]; then
        log "[merge_$name] --quick：跳过真实合并（会写约 6GB），改跑 merge_model.py --dry-run"
        cmd+=(--dry-run)
    fi

    log "[merge_$name] 日志=$logfile"
    printf '[step2] 命令: '; printf '%q ' "${cmd[@]}"; printf '\n'
    if [ "$DRY_RUN" -eq 1 ]; then
        log "[dry-run] 不执行。"
        return 0
    fi
    ( cd "$REPO" && "${cmd[@]}" ) 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        log "[FAIL] merge_$name 退出码 $rc，见 $logfile"
        return "$rc"
    fi
    log "[merge_$name] 合并完成"
    return 0
}

# ---------------------------------------------------------------------------
# 主流程：SFT → 合并 → DPO → 合并
# ---------------------------------------------------------------------------
log "REPO      = $REPO"
log "PYTHON    = $PYTHON"
log "stage=$STAGE  dry_run=$DRY_RUN  quick=$QUICK  max_retries=$MAX_RETRIES"
log "run_id    = $RUN_ID"

# 数据没准备好就别开训 —— 一次 SFT 要几十分钟，让它跑到读数据那一步才失败
# 是纯粹的浪费。dataset_info.json 是 step1 的最后一个产物，它在 = 数据齐了。
if [ ! -f "$DATA_DIR/dataset_info.json" ]; then
    log "[FAIL] 找不到 $DATA_DIR/dataset_info.json"
    log "       请先跑： \"\$PYTHON\" Week8/scripts/step1_data_prep.py"
    exit 1
fi

if [ "$STAGE" = "sft" ] || [ "$STAGE" = "all" ]; then
    log "===== 阶段 1/4：SFT（Week3 冠军超参 r=32/α=64/lr=1e-4/3ep）====="
    run_train_stage sft "$SFT_CONFIG" || exit 1
    log "===== 阶段 2/4：合并 SFT adapter → $MERGED_SFT ====="
    run_merge_stage sft "$MERGE_SFT_CONFIG" || exit 1
fi

if [ "$STAGE" = "dpo" ] || [ "$STAGE" = "all" ]; then
    # 单独跑 dpo 时 policy 可能还不存在（--dry-run / --quick 下不算错）。
    if [ ! -d "$MERGED_SFT" ] && [ "$DRY_RUN" -eq 0 ] && [ "$QUICK" -eq 0 ]; then
        log "[FAIL] DPO 的 policy 起点不存在：$MERGED_SFT"
        log "       它是 SFT 合并的产物。先跑 --stage sft，或用 --stage all。"
        exit 1
    fi
    log "===== 阶段 3/4：DPO（Week4 冠军超参 β=0.5/lr=5e-6/2ep）====="
    run_train_stage dpo "$DPO_CONFIG" || exit 1
    log "===== 阶段 4/4：合并 DPO adapter → $MERGED_DPO ====="
    run_merge_stage dpo "$MERGE_DPO_CONFIG" || exit 1
fi

log "全部完成。日志目录: $LOG_DIR"
[ -f "$RETRY_HISTORY" ] && log "重试历史: $RETRY_HISTORY"
exit 0
