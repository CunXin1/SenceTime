#!/usr/bin/env bash
# ============================================================================
#  step4_deploy.sh — Week8 Day41 / 任务书 41.2
#  自动拉起 vLLM 服务（加载量化模型）+ Gradio 界面（后台运行），并检查健康状态。
#  Starts the vLLM server (quantized model) and the Gradio UI in the
#  background, then health-checks both.
#
#  用法 / Usage:
#    bash Week8/scripts/step4_deploy.sh                # 起 awq 后端 + Gradio
#    bash Week8/scripts/step4_deploy.sh --variant gptq
#    bash Week8/scripts/step4_deploy.sh --variant vl --port 8001 --no-ui
#    bash Week8/scripts/step4_deploy.sh --status       # 只查健康，不启动
#    bash Week8/scripts/step4_deploy.sh --stop         # 停掉本脚本起的服务
#    bash Week8/scripts/step4_deploy.sh --dry-run      # 只打印将要执行的命令
#
#  ---------------------------------------------------------------------------
#  ★ 取舍一：这个脚本必须跨过 Windows / WSL 的边界，因为两个服务**不在同一个系统里**
#    vLLM 没有 Windows 轮子——第 7 周实测过，官方只发 Linux wheel，
#    所以推理服务只能跑在 WSL2 的 ~/venvs/vllm 里。
#    而 Gradio 前端跑在 Windows 侧的 .venv（第 5 周的多模态资源、第 3 周的题集
#    都在 Windows 文件系统上，来回跨 /mnt/c 反而慢）。
#    于是本脚本做的事是：**在哪边就用哪边的启动方式**——
#      · 脚本本身跑在 WSL/Linux  → 直接 nohup 起 vllm
#      · 脚本本身跑在 Git Bash   → 用 `wsl.exe -e bash -lc` 把命令投递进 WSL
#    `detect_side()` 靠 /proc/version 里的 "microsoft" 字样和 $WSL_DISTRO_NAME
#    判断，比看 uname 可靠（Git Bash 的 uname 是 MINGW64_NT，WSL 是 Linux）。
#    The two services live in different OSes; this script dispatches to
#    whichever launcher matches the side it is running on.
#
#  ★ 取舍二：健康检查为什么必须用 curl 打 /v1/models，而不是"看进程还在不在"
#    vLLM 从进程起来到能接请求要 40~120 秒（加载权重 + 预分配 KV cache +
#    捕获 CUDA graph）。这段时间里进程活得好好的，但任何请求都会连接被拒。
#    如果 step4 起完就报"部署成功"，紧接着的冒烟请求必然失败，
#    而失败原因看起来像"服务挂了"，实际只是没等它起完。
#    所以这里轮询 `GET /v1/models` 直到返回 200 或超时（--timeout，默认 300s），
#    并且**把等待过程打出来**——让人看见它在等，而不是以为卡死了。
#    Poll the HTTP endpoint, not the process: vLLM needs 40-120s before it
#    can accept requests, and a live PID means nothing during that window.
#
#  ★ 取舍三：curl 必须带 --noproxy
#    第 7 周踩过整整一轮：Windows 注册表里配了系统代理，httpx / curl 都会读
#    环境里的 http_proxy 并把发往 127.0.0.1 的请求也塞进代理，代理转不了
#    localhost，回 502。症状极具迷惑性——服务是好的，请求没到。
#    Gradio 那边靠 app.py 里的 NO_PROXY + `httpx.Client(trust_env=False)` 解决，
#    这里健康检查是 curl，对应的开关是 `--noproxy '*'`。
#    Always --noproxy: a system proxy will hijack 127.0.0.1 and return 502.
#
#  ★ 取舍四：为什么默认后端是 awq 而不是 fp16
#    任务书 41.2 明确写"加载量化模型"。而且第 7 周的实测数据支持这个默认值：
#    AWQ 权重显存 1.95 GiB（比 FP16 的 5.79 少 66.3%），b=1 吞吐 226.5 tok/s
#    （FP16 是 119.8），代价是 PPL +7.8%。省下来的显存全部变成 KV cache，
#    32K 上下文下最大并发从 11.76× 提到 16.12×。部署场景要的就是这个。
#
#  ★ 取舍五：PID 文件写在 Week8/logs/ 而不是 /tmp
#    /tmp 在 Windows 和 WSL 里是两个目录，脚本又要跨边界，用 /tmp 必然对不上。
#    Week8/logs/ 两边都能通过 /mnt/c 或盘符看到，`--stop` 才能真的停掉。
#    ★ 但 WSL 侧的 vLLM PID 是 **WSL 命名空间里的 PID**，Windows 的 taskkill
#      杀不掉它——所以 stop 的时候同样要把 kill 投递进 WSL 去执行。
# ============================================================================
set -uo pipefail                  # 不加 -e：健康检查失败时要自己收尾并给建议

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$REPO/Week8/configs/pipeline.env"

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
VARIANT="awq"
VLLM_PORT=8000
UI_PORT=7860
TIMEOUT=300
NO_UI=0
DRY_RUN=0
ACTION="up"

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --variant)   VARIANT="${2:-}"; shift 2 ;;
        --variant=*) VARIANT="${1#*=}"; shift ;;
        --port)      VLLM_PORT="${2:-}"; shift 2 ;;
        --port=*)    VLLM_PORT="${1#*=}"; shift ;;
        --ui-port)   UI_PORT="${2:-}"; shift 2 ;;
        --ui-port=*) UI_PORT="${1#*=}"; shift ;;
        --timeout)   TIMEOUT="${2:-}"; shift 2 ;;
        --timeout=*) TIMEOUT="${1#*=}"; shift ;;
        --no-ui)     NO_UI=1; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        --status)    ACTION="status"; shift ;;
        --stop)      ACTION="down"; shift ;;
        -h|--help)   usage 0 ;;
        *) echo "[FAIL] 未知参数: $1"; usage 1 ;;
    esac
done

case "$VARIANT" in fp16|awq|gptq|vl) ;; *)
    echo "[FAIL] --variant 只能是 fp16 / awq / gptq / vl，收到: $VARIANT"; exit 1 ;;
esac
# vl 是多模态槽位，按第 7 周的约定走 8001（app.py 的 BACKENDS 里写死了）
if [ "$VARIANT" = "vl" ] && [ "$VLLM_PORT" = "8000" ]; then
    VLLM_PORT=8001
fi

DEPLOY_LOG_DIR="$LOG_DIR"
mkdir -p "$DEPLOY_LOG_DIR"
VLLM_PID_FILE="$DEPLOY_LOG_DIR/vllm_${VARIANT}.pid"
UI_PID_FILE="$DEPLOY_LOG_DIR/gradio_${UI_PORT}.pid"
VLLM_LOG="$DEPLOY_LOG_DIR/deploy_vllm_${VARIANT}.log"
UI_LOG="$DEPLOY_LOG_DIR/deploy_gradio_${UI_PORT}.log"

log() { echo "[step4][$(date +%H:%M:%S)] $*"; }
run() { if [ "$DRY_RUN" = "1" ]; then echo "  (dry-run) $*"; else eval "$@"; fi; }

# ---------------------------------------------------------------------------
# 我在哪一侧？（见 ★取舍一）
# ---------------------------------------------------------------------------
detect_side() {
    # ★ 判断顺序不能反：Git Bash(MSYS2) **也有** /proc/version，内容是
    #   "MINGW64_NT-10.0-26200 version 3.6.9-..."（2026-08-25 实测）。
    #   所以必须先用 uname -s 把 MINGW/MSYS/CYGWIN 挑出去，再拿
    #   /proc/version 里的 "microsoft" 认 WSL。第一版把"/proc/version 存在"
    #   当成"我在 Linux 上"，于是在 Git Bash 里判成 linux，
    #   直接去 nohup 一个 Windows 上根本不存在的 vllm。
    #   MSYS2 also provides /proc/version; check uname first.
    case "$(uname -s 2>/dev/null)" in
        MINGW*|MSYS*|CYGWIN*) echo "windows"; return ;;
    esac
    if [ -n "${WSL_DISTRO_NAME:-}" ]; then
        echo "wsl"
    elif [ -r /proc/version ] && grep -qi microsoft /proc/version 2>/dev/null; then
        echo "wsl"
    else
        echo "linux"                    # 原生 Linux
    fi
}
SIDE="$(detect_side)"

# 把仓库路径翻译成 WSL 能认的形式。REPO 在 Git Bash 下是 "C:/Users/..."
# （pipeline.env 里 cygpath -m 转过），WSL 里必须是 "/mnt/c/Users/..."。
to_wsl_path() {
    printf '%s' "$1" | sed -E 's#^([A-Za-z]):#/mnt/\L\1#; s#\\#/#g'
}

# ---------------------------------------------------------------------------
# 健康检查（见 ★取舍二、三）
# ---------------------------------------------------------------------------
CURL="$(command -v curl || true)"

http_code() {
    # $1 = url。返回 HTTP 状态码；连不上返回 000。
    # ★ 末尾必须是 `; true`，不能写成 `|| echo "000"`：
    #   连不上时 curl **既**把 "000" 打到 stdout（-w '%{http_code}' 照样输出），
    #   **又**返回非零退出码。写成 `|| echo "000"` 会让两者都发生，函数返回
    #   "000000"，后面所有 `[ "$code" = "200" ]` 的比较全部失效——而且失效得
    #   很安静：状态永远显示 ❌，看起来像"服务没起来"。
    #   （2026-08-25 实测：--status 打出 HTTP=000000）
    #   curl prints 000 AND exits non-zero; `|| echo 000` yields "000000".
    [ -z "$CURL" ] && { echo "000"; return; }
    "$CURL" -s -o /dev/null -w '%{http_code}' --noproxy '*' \
            --max-time 5 "$1" 2>/dev/null; true
}

wait_healthy() {
    # $1 = url  $2 = 名字  $3 = 超时秒数
    local url="$1" name="$2" limit="$3" waited=0 code
    log "等待 $name 就绪（最多 ${limit}s）：$url"
    while [ "$waited" -lt "$limit" ]; do
        code="$(http_code "$url")"
        if [ "$code" = "200" ]; then
            log "  ✅ $name 就绪（${waited}s，HTTP 200）"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        # 每 15 秒打一次心跳：让人看见它在等，而不是以为脚本卡死了
        if [ $((waited % 15)) -eq 0 ]; then
            log "  … ${waited}s HTTP=$code"
        fi
    done
    log "  ❌ $name 在 ${limit}s 内未就绪（最后 HTTP=$code）"
    return 1
}

show_status() {
    local code_v code_u
    code_v="$(http_code "http://127.0.0.1:$VLLM_PORT/v1/models")"
    code_u="$(http_code "http://127.0.0.1:$UI_PORT/")"
    echo "---------------------------------------------------------------"
    echo "  vLLM   http://127.0.0.1:$VLLM_PORT/v1/models   HTTP=$code_v  $([ "$code_v" = 200 ] && echo ✅ || echo ❌)"
    echo "  Gradio http://127.0.0.1:$UI_PORT/              HTTP=$code_u  $([ "$code_u" = 200 ] && echo ✅ || echo ❌)"
    if [ "$code_v" = "200" ] && [ -n "$CURL" ]; then
        echo "  已加载模型：$("$CURL" -s --noproxy '*' --max-time 5 \
            "http://127.0.0.1:$VLLM_PORT/v1/models" \
            | sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | head -1)"
    fi
    echo "  日志：$VLLM_LOG"
    echo "        $UI_LOG"
    echo "---------------------------------------------------------------"
    [ "$code_v" = "200" ]
}

# ---------------------------------------------------------------------------
# 启动 vLLM
# ---------------------------------------------------------------------------
start_vllm() {
    if [ "$(http_code "http://127.0.0.1:$VLLM_PORT/v1/models")" = "200" ]; then
        log "vLLM 已在 :$VLLM_PORT 上运行，跳过启动"
        return 0
    fi

    # 复用第 7 周的 serve_vllm.sh —— 那里面有 WSL2 的两个必开环境变量
    # （VLLM_WSL2_ENABLE_PIN_MEMORY / VLLM_USE_FLASHINFER_SAMPLER）和
    # 每种精度的模型路径。在这里重写一遍等于把踩过的坑再埋一次。
    local serve_rel="Week7/code/serve_vllm.sh"

    if [ "$SIDE" = "windows" ]; then
        local wrepo; wrepo="$(to_wsl_path "$REPO")"
        log "在 WSL 中启动 vLLM（variant=$VARIANT, port=$VLLM_PORT）"
        # 单引号里再拼路径：仓库路径含空格和撇号，必须整体引起来。
        local inner
        inner="source ~/venvs/vllm/bin/activate && cd \"$wrepo\" && \
nohup bash $serve_rel $VARIANT $VLLM_PORT > \"$(to_wsl_path "$VLLM_LOG")\" 2>&1 & echo \$!"
        if [ "$DRY_RUN" = "1" ]; then
            # ★ 这里故意**不**给 $inner 套单引号来展示。仓库路径里有一个撇号
            #   （Ruibo's Desktop），套上单引号打印出来会变成一条自己都跑不了的
            #   命令，照着复制粘贴的人必然踩坑。真正执行时走的是
            #   `wsl.exe -e bash -lc "$inner"`——bash 把整串作为**一个 argv**
            #   传进去，而撇号在 $inner 内部是包在双引号里的，WSL 侧解析正确
            #   （已实测：cd 进带撇号的目录并成功 import vllm 0.27.1）。
            echo "  (dry-run) wsl.exe -e bash -lc <<单条命令>>，内容："
            echo "            $inner"
            return 0
        fi
        local pid
        pid="$(wsl.exe -e bash -lc "$inner" 2>/dev/null | tr -d '\r')"
        echo "$pid" > "$VLLM_PID_FILE"
        log "vLLM 已投递进 WSL，WSL 内 PID=$pid（记于 $VLLM_PID_FILE）"
    else
        log "本地启动 vLLM（variant=$VARIANT, port=$VLLM_PORT）"
        if [ "$DRY_RUN" = "1" ]; then
            echo "  (dry-run) nohup bash $REPO/$serve_rel $VARIANT $VLLM_PORT > $VLLM_LOG 2>&1 &"
            return 0
        fi
        # shellcheck disable=SC2086
        nohup bash "$REPO/$serve_rel" "$VARIANT" "$VLLM_PORT" > "$VLLM_LOG" 2>&1 &
        echo $! > "$VLLM_PID_FILE"
        log "vLLM PID=$(cat "$VLLM_PID_FILE")"
    fi
}

# ---------------------------------------------------------------------------
# 启动 Gradio（永远在 Windows 侧 / 脚本所在侧的 .venv 里）
# ---------------------------------------------------------------------------
start_ui() {
    [ "$NO_UI" = "1" ] && { log "--no-ui：跳过 Gradio"; return 0; }
    if [ "$(http_code "http://127.0.0.1:$UI_PORT/")" = "200" ]; then
        log "Gradio 已在 :$UI_PORT 上运行，跳过启动"
        return 0
    fi
    local app="$REPO/Week7/code/app.py"
    if [ ! -f "$app" ]; then
        log "❌ 找不到 $app —— Gradio 前端是第 7 周 Day37 的产物"
        return 1
    fi
    log "启动 Gradio（port=$UI_PORT）"
    if [ "$DRY_RUN" = "1" ]; then
        echo "  (dry-run) nohup \"$PYTHON\" \"$app\" --port $UI_PORT > $UI_LOG 2>&1 &"
        return 0
    fi
    nohup "$PYTHON" "$app" --port "$UI_PORT" > "$UI_LOG" 2>&1 &
    echo $! > "$UI_PID_FILE"
    log "Gradio PID=$(cat "$UI_PID_FILE")"
}

# ---------------------------------------------------------------------------
# 停止（见 ★取舍五：WSL 侧的 PID 要投递回 WSL 去杀）
# ---------------------------------------------------------------------------
stop_all() {
    if [ -f "$VLLM_PID_FILE" ]; then
        local pid; pid="$(cat "$VLLM_PID_FILE")"
        if [ "$SIDE" = "windows" ]; then
            log "在 WSL 中停止 vLLM（PID=$pid）"
            run "wsl.exe -e bash -lc 'kill $pid 2>/dev/null; sleep 3; kill -9 $pid 2>/dev/null; true'"
        else
            log "停止 vLLM（PID=$pid）"
            run "kill $pid 2>/dev/null; sleep 3; kill -9 $pid 2>/dev/null; true"
        fi
        [ "$DRY_RUN" = "1" ] || rm -f "$VLLM_PID_FILE"
    else
        log "没有 vLLM 的 PID 文件，跳过（服务可能是别处起的，本脚本不越权去杀）"
    fi

    if [ -f "$UI_PID_FILE" ]; then
        local pid; pid="$(cat "$UI_PID_FILE")"
        log "停止 Gradio（PID=$pid）"
        run "kill $pid 2>/dev/null; true"
        [ "$DRY_RUN" = "1" ] || rm -f "$UI_PID_FILE"
    fi
    log "已停止。显存释放需要几秒，可用 nvidia-smi 确认。"
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
log "side=$SIDE  variant=$VARIANT  vllm_port=$VLLM_PORT  ui_port=$UI_PORT"

case "$ACTION" in
    status) show_status; exit $? ;;
    down)   stop_all; exit 0 ;;
esac

if [ -z "$CURL" ]; then
    log "⚠️  找不到 curl，健康检查将无法进行（服务仍会被启动）。"
    log "    Git Bash 自带 curl；WSL 里 apt install curl。"
fi

start_vllm || exit 1
if [ "$DRY_RUN" = "1" ]; then
    start_ui
    log "--dry-run 结束：以上是将要执行的命令，未真正启动任何进程。"
    exit 0
fi

if ! wait_healthy "http://127.0.0.1:$VLLM_PORT/v1/models" "vLLM" "$TIMEOUT"; then
    echo
    log "vLLM 启动失败，日志最后 30 行："
    tail -30 "$VLLM_LOG" 2>/dev/null || echo "  （日志文件都没生成，八成是 WSL 里的 venv 路径不对）"
    echo
    log "常见原因："
    log "  1) 显存不够：先 nvidia-smi 看有没有别的进程占卡（训练/评测/游戏）"
    log "  2) 模型目录不存在：awq/gptq 模型是第 7 周 Day34/35 量化出来的"
    log "  3) WSL 里 ~/venvs/vllm 不存在：见 Week7/code/setup_wsl_vllm.sh"
    exit 1
fi

start_ui
if [ "$NO_UI" != "1" ]; then
    # Gradio 起得快（几秒），但仍然要等——它启动时会自己请求一次本地地址，
    # 那一步在有系统代理的机器上会 502（app.py 里用 NO_PROXY 解决了）。
    wait_healthy "http://127.0.0.1:$UI_PORT/" "Gradio" 60 || \
        log "⚠️  Gradio 未就绪，但 vLLM 是好的——可以先用 curl / client_demo.py 打接口"
fi

echo
show_status
echo
log "部署完成。停止服务：bash Week8/scripts/step4_deploy.sh --variant $VARIANT --stop"
