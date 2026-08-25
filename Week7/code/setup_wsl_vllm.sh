#!/usr/bin/env bash
# ============================================================================
#  setup_wsl_vllm.sh — Week7 Day34
#  在 WSL2 Ubuntu 里搭建量化环境与 vLLM 服务环境。
#  Set up the quantization env and the vLLM serving env inside WSL2 Ubuntu.
#
#  运行 / Run（在 WSL 里，重启完成并装好发行版之后）:
#    cd /mnt/c/Users/Ruibo\'s\ Desktop/SenceTime_Weeks1-5
#    bash Week7/code/setup_wsl_vllm.sh 2>&1 | tee Week7/deliverables/logs/setup_wsl.log
#
#  ★ 为什么要建两个 venv（~/venvs/quant 与 ~/venvs/vllm）
#    vLLM 会把 torch 钉在它自己编译时用的那个精确版本，而 gptqmodel / llm-compressor /
#    LLaMA-Factory 各自对 transformers 和 torch 有独立诉求。装一个 venv 里，pip 的
#    依赖求解迟早会把 vLLM 的 torch 换掉，换完 vLLM 的 CUDA 扩展就加载不了（.so 与
#    torch ABI 绑定）。而量化和服务本来就是**先后两个阶段**，没有共享运行时的必要——
#    沿用你 Week5 的 .venv-vlm / Week6 的 .venv-agent 同一套隔离思路。
#    Quantization and serving are sequential phases; isolating them avoids pip
#    resolving away vLLM's pinned torch (its CUDA extensions are ABI-bound to it).
#
#  ★ 为什么不 pin 死版本号
#    Week1-6 的环境要可复现，是因为那些是**训练**脚本，版本漂移会改变数值结果。
#    本周是推理部署，且 vLLM/llm-compressor 迭代极快、跨版本 API 变动大，写死一个
#    我没实测过的版本反而更容易一开始就装不上。做法改成：让 pip 解最新的可用组合，
#    装完立刻 pip freeze 存档到 deliverables/，用**事后记录**换取可复现性。
#    Versions are recorded post-hoc via pip freeze rather than pinned blindly.
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGDIR="$REPO/Week7/deliverables"
mkdir -p "$LOGDIR/logs"

say() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
say "0/5 前置检查：GPU 直通"
# WSL 用的是 Windows 侧的 NVIDIA 驱动（610.88），**不要**在 WSL 里再装驱动——
# 装了会覆盖 /usr/lib/wsl/lib 下的直通库，反而把 GPU 弄丢。
# Do NOT install an NVIDIA driver inside WSL; it shadows the passthrough libs.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi 不在 PATH，尝试 /usr/lib/wsl/lib ..."
    export PATH="$PATH:/usr/lib/wsl/lib"
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv \
    || { echo "[FAIL] GPU 未直通。检查 Windows 侧驱动，以及是否已重启。"; exit 1; }

# ---------------------------------------------------------------------------
say "1/5 系统依赖"
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12-venv python3-pip build-essential git

# ---------------------------------------------------------------------------
say "2/5 建 vLLM 服务环境 ~/venvs/vllm"
python3 -m venv ~/venvs/vllm
# shellcheck disable=SC1090
source ~/venvs/vllm/bin/activate
pip install -q --upgrade pip
# vllm 自带匹配的 torch；openai SDK 用于 Day36 的客户端与 Day35 的评测脚本。
pip install vllm openai
python - <<'PY'
import torch, vllm
print(f"vllm={vllm.__version__}  torch={torch.__version__}  cuda_ok={torch.cuda.is_available()}")
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
PY
pip freeze > "$LOGDIR/wsl_env_vllm_freeze.txt"
deactivate

# ---------------------------------------------------------------------------
say "3/5 建量化环境 ~/venvs/quant"
python3 -m venv ~/venvs/quant
# shellcheck disable=SC1090
source ~/venvs/quant/bin/activate
pip install -q --upgrade pip
# AWQ 走 llm-compressor：AutoAWQ 仓库已归档停止维护，官方指向 llm-compressor，
# 且它产出的 compressed-tensors 格式能被 vLLM 直接加载，与 GPTQ 共用一套工具链。
# AWQ via llm-compressor (AutoAWQ is archived); GPTQ via LF + gptqmodel.
pip install llmcompressor
pip install "gptqmodel>=2.0.0" "optimum>=1.24.0"
# LLaMA-Factory 用 --no-deps 装，否则 pip 会连带降级 transformers/accelerate/datasets
# ——这是你 Week5 setup_venv_vlm.ps1 里已经踩过并固化的解法，这里沿用。
pip install -e "$REPO/LLaMA-Factory" --no-deps
# 补齐 LF 自身缺的那几个轻量依赖（不碰 transformers 系）
pip install omegaconf fire tyro sse-starlette fastapi uvicorn
pip freeze > "$LOGDIR/wsl_env_quant_freeze.txt"
deactivate

# ---------------------------------------------------------------------------
say "4/5 关键能力自检"
source ~/venvs/quant/bin/activate
python - <<'PY'
import importlib
for m in ("llmcompressor", "gptqmodel", "llamafactory", "transformers"):
    try:
        mod = importlib.import_module(m)
        print(f"  [ok] {m} = {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  [FAIL] {m}: {type(e).__name__}: {e}")
PY
deactivate

# ★ 这一步是 Day38 的关键前置：Gemma-4-E4B 用的是 PLE / per-layer-embedding 那一系
#   架构，这类模型进 vLLM 的 registry 往往比 transformers 晚好几个月。现在就查清楚，
#   免得 Day38 才发现要临时改方案。
#   Check now whether Gemma-4 is in vLLM's registry — its architecture family
#   historically lags transformers support by months.
say "5/5 vLLM 模型架构支持自检（Day38 前置）"
source ~/venvs/vllm/bin/activate
python - <<'PY'
from vllm.model_executor.models.registry import ModelRegistry
archs = sorted(ModelRegistry.get_supported_archs())
print(f"vLLM 已注册架构 {len(archs)} 个")
for kw in ("Qwen2_5_VL", "Qwen2VL", "Gemma"):
    hit = [a for a in archs if kw.lower().replace("_", "") in a.lower().replace("_", "")]
    print(f"  {kw:12s} -> {hit if hit else '★ 未注册，需走 HF transformers 后端'}")
PY
deactivate

say "完成"
cat <<'EOF'
下一步：
  1) 量化 AWQ :  source ~/venvs/quant/bin/activate
                 python Week7/code/quantize_awq.py
  2) 量化 GPTQ:  python -m llamafactory.cli export Week7/configs/export_gptq_w4.yaml
  3) 起服务   :  source ~/venvs/vllm/bin/activate
                 bash Week7/code/serve_vllm.sh awq
环境版本已存档到 Week7/deliverables/wsl_env_{vllm,quant}_freeze.txt
EOF
