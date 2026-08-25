#!/usr/bin/env bash
# ============================================================================
#  serve_vllm.sh — Week7 Day36
#  起一个 OpenAI 兼容的 vLLM 服务，三种精度共用这一个脚本。
#  Launch an OpenAI-compatible vLLM server; one script for all three precisions.
#
#  运行 / Run（WSL 的 vllm 环境）:
#    source ~/venvs/vllm/bin/activate
#    cd /mnt/c/Users/Ruibo\'s\ Desktop/SenceTime_Weeks1-5
#    bash Week7/code/serve_vllm.sh fp16      # 或 awq / gptq / vl
#
#  ★ --gpu-memory-utilization 为什么必须三种精度都设成同一个值
#    vLLM 启动时会按这个比例**预分配** KV cache，把显存一次性吃满。所以不管
#    FP16 还是 4-bit，nvidia-smi 看到的进程显存都是同一个数——直接用 nvidia-smi
#    量"量化省了多少显存"必然得出"没省"的错误结论。
#    正确口径是看启动日志里的模型权重占用（bench_quant.py 会解析），而把
#    utilization 固定住，是为了让三次测量的 KV cache 预算相同、吞吐可比。
#    Fixed utilization keeps the KV-cache budget identical across runs so the
#    throughput numbers are comparable; weight size comes from the startup log.
#
#  ★ --served-model-name 为什么统一成 qwen3b
#    Gradio 前端和 bench 脚本按这个名字调用。三种精度分别起服务时都叫同一个名字，
#    切换后端不用改客户端代码（任务书 36.1 要求的就是这个参数）。
#    量化种类靠端口区分，日志里也会记，不会混淆。
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VARIANT="${1:-awq}"
PORT="${2:-8000}"
UTIL="${GPU_UTIL:-0.85}"          # 三种精度必须一致，见上方说明
LOGDIR="$REPO/Week7/deliverables/logs"
mkdir -p "$LOGDIR"

# ★ WSL2 必须开这个，否则 vLLM 0.27.1 起不来（2026-08-21 实测踩坑）
#   报错是 `RuntimeError: UVA is not available`，链条要追三层才看得清：
#     1) vLLM 0.27.1 默认使用 V2 model runner（gpu/model_runner.py）
#     2) V2 runner 的 RequestState 用 UvaBuffer 做 CPU<->GPU 的暂存
#     3) buffer_utils.py:47 检查 is_uva_available()，而它的实现就是
#        `is_pin_memory_available() or is_cpu()`（utils/platform_utils.py:57）
#   而 vLLM **在 WSL2 上默认关闭 pinned memory**——envs.py 的注释写得很清楚：
#   内核 >= 4.19.121 其实支持，只是有小幅性能回退，所以默认关，需要 UVA 时手动开。
#   本机内核 6.18.33.2，远高于门槛，直接开。
#   注意：这个"小幅性能回退"对 fp16/awq/gptq 三次测量是**同等施加**的，
#   不影响 Day35 三方对比的相对结论，但绝对吞吐值不应拿去和原生 Linux 的数字比。
#   Without this, vLLM's V2 model runner fails on WSL2 since pinned memory
#   (which UVA depends on) is disabled by default there.
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

# ★ 关掉 FlashInfer 采样器，否则启动到最后一步崩（2026-08-21 实测踩坑之二）
#   报错：RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
#   注意别被 nvcc 误导成"要装 CUDA toolkit"。分清两件事：
#     · **注意力**后端选的是 FLASH_ATTN，vLLM wheel 里自带预编译 kernel，没问题
#       （日志：Using FLASH_ATTN attention backend out of potential backends:
#        ['FLASH_ATTN','FLASHINFER','TRITON_ATTN','FLEX_ATTENTION']）
#     · 崩的是**采样器**：topk_topp_sampler.py 默认走 FlashInfer 的
#       top_k_top_p_sampling_from_logits，而 flashinfer 那条路径是 **JIT 编译**的，
#       运行时才现场 nvcc 编 kernel —— WSL 里只有直通的运行时驱动，没有 toolkit。
#   为装 3GB 的 CUDA toolkit 去换一个采样器的边际加速不划算（C 盘也紧张），
#   直接退回 PyTorch 原生采样。对本周影响为零：bench 全程 temperature=0 走贪心，
#   Gradio 那边的 top-p/top-k 用原生实现同样正确，只是极端高并发下略慢。
#   The failure is in the SAMPLER (JIT-compiled), not the attention backend.
export VLLM_USE_FLASHINFER_SAMPLER=0

case "$VARIANT" in
  fp16) MODEL="$REPO/models/Qwen2.5-3B-week4-dpo-merged";     NAME="qwen3b"; EXTRA=(--dtype float16) ;;
  awq)  MODEL="$REPO/models/Qwen2.5-3B-week4-dpo-awq-w4";     NAME="qwen3b"; EXTRA=() ;;
  gptq) MODEL="$REPO/models/Qwen2.5-3B-week4-dpo-gptq-w4";    NAME="qwen3b"; EXTRA=() ;;
  # Day38 的多模态槽位。--limit-mm-per-prompt 限制每轮最多 1 张图：Qwen2.5-VL 的视觉
  # token 数随分辨率动态变化，不设上限时一张大图能吃掉几千 token 的 KV cache。
  vl)   MODEL="$REPO/models/Qwen2.5-VL-7B-Instruct";          NAME="qwen-vl"
        EXTRA=(--limit-mm-per-prompt '{"image":1}' --max-model-len 8192) ;;
  *)    echo "用法: bash serve_vllm.sh {fp16|awq|gptq|vl} [port]"; exit 1 ;;
esac

if [ ! -d "$MODEL" ]; then
    echo "[FAIL] 模型不存在: $MODEL"
    echo "       fp16 需先跑通 Week4；awq/gptq 见 Week7/README.md 第三节；vl 需先下载。"
    exit 1
fi

LOG="$LOGDIR/serve_${VARIANT}.log"
echo "[serve] variant=$VARIANT  port=$PORT  util=$UTIL"
echo "[serve] model=$MODEL"
echo "[serve] log=$LOG   (bench_quant.py 会从这个日志里解析权重显存)"

# --host 0.0.0.0 是必须的：Gradio 跑在 Windows 侧，靠 WSL2 的 localhost 转发访问。
# 只绑 127.0.0.1 的话 Windows 连不上。
exec vllm serve "$MODEL" \
    --served-model-name "$NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --gpu-memory-utilization "$UTIL" \
    --max-num-seqs 32 \
    "${EXTRA[@]}" 2>&1 | tee "$LOG"
