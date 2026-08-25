#!/usr/bin/env bash
# ============================================================================
#  run_bench_all.sh — Week7 Day35
#  自动跑完 fp16 / awq / gptq 三轮「起服务 → 测 → 停服务」，最后汇总成对比表。
#  Drive the full three-precision benchmark end to end.
#
#  ★ 为什么值得脚本化
#    三种精度必须**串行**跑：24GB 显存塞不下两个服务，而且 --gpu-memory-utilization
#    是按「整卡空闲」算的，并行会让 KV cache 预算互相污染，吞吐数字失去可比性。
#    手工编排要来回切六次终端、每次等模型加载，既慢又容易漏掉 kill 导致下一轮
#    OOM 或端口占用——那种错误还不会立刻报错，而是让结果悄悄失真。
#    Serial by necessity: parallel servers would contaminate each other's KV-cache
#    budget and silently skew the throughput numbers.
#
#  运行 / Run（WSL）:
#    bash Week7/code/run_bench_all.sh 2>&1 | tee Week7/deliverables/logs/bench_all.log
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${PORT:-8000}"
VARIANTS="${*:-fp16 awq gptq}"

say() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

stop_server() {
    pkill -f "vllm serve" 2>/dev/null
    sleep 8
    pkill -9 -f "vllm serve" 2>/dev/null
    sleep 3
}

wait_ready() {
    # vLLM 冷启动要几分钟：模型在 /mnt/c（drvfs）上读得慢，外加 torch.compile。
    for _ in $(seq 1 60); do
        curl -s -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && return 0
        sleep 10
    done
    return 1
}

for v in $VARIANTS; do
    say "[$v] 1/3 起服务"
    stop_server
    source ~/venvs/vllm/bin/activate
    bash "$REPO/Week7/code/serve_vllm.sh" "$v" "$PORT" &
    SERVE_PID=$!

    say "[$v] 2/3 等就绪"
    if ! wait_ready; then
        echo "[FAIL] $v 服务未就绪，看 Week7/deliverables/logs/serve_${v}.log"
        kill $SERVE_PID 2>/dev/null
        continue
    fi
    echo "[ok] $v 已就绪"

    say "[$v] 3/3 跑 bench"
    python "$REPO/Week7/code/bench_quant.py" --variant "$v" --port "$PORT"
done

stop_server
say "汇总"
source ~/venvs/vllm/bin/activate
python "$REPO/Week7/code/bench_quant.py" --report
