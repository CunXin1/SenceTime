#!/usr/bin/env bash
# ============================================================================
#  setup_wsl_lf.sh — Week7 Day35（2026-08-21 追加）
#  给 LLaMA-Factory 单开第三个 venv，专用于 GPTQ 导出。
#  A third venv dedicated to LLaMA-Factory's GPTQ export.
#
#  ★ 为什么非要再开一个环境（不是洁癖，是撞上了真的 API 断裂）
#    setup_wsl_vllm.sh 建的 ~/venvs/quant 里，pip 为 llm-compressor 解出的是
#    transformers 5.14.1 / trl 1.10.0 / peft 0.20.0 / datasets 5.0.1。
#    而 LF 0.9.6.dev0 声明的区间是：
#        transformers <=5.7.0   datasets <=4.0.0   peft <=0.18.1
#        trl          <=0.24.0  tyro     <0.9.0
#    先试了最便宜的 DISABLE_VERSION_CHECK=1 跳过 LF 的 check_dependencies()，
#    结果在 model/loader.py:28 直接炸：
#        ImportError: cannot import name 'AutoModelForCausalLMWithValueHead' from 'trl'
#    ——trl 1.x 已经把这个类删了。这说明那不是"保守的版本声明"，是真实的 API 断裂，
#    绕不过去。而反过来把 quant 环境降级到 LF 的区间同样不可接受：AWQ 模型是用
#    transformers 5.14.1 产出的，降级会让 wsl_env_quant_freeze.txt 与实际产出模型的
#    环境对不上，破坏可复现性。
#    结论：AWQ 与 GPTQ 是两条独立的工具链，各自钉住各自的依赖，互不干扰。
#    Downgrading the quant env would invalidate the freeze that produced the AWQ
#    model; trl 1.x genuinely removed the class LF imports. Hence a third venv.
#
#  运行 / Run:
#    bash Week7/code/setup_wsl_lf.sh 2>&1 | tee Week7/deliverables/logs/setup_lf.log
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGDIR="$REPO/Week7/deliverables"
mkdir -p "$LOGDIR/logs"

say() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

say "1/3 建 ~/venvs/lf"
python3 -m venv ~/venvs/lf
# shellcheck disable=SC1090
source ~/venvs/lf/bin/activate
pip install -q --upgrade pip

say "2/3 依赖：LF 区间 + 一处**经实测的**上界突破"
#  ★ 三方夹击，以及为什么最终只突破 transformers 这一条
#    第一版本来是把五个依赖全钉在 LF 声明的上界（transformers==5.7.0 等），装完
#    自检发现反向撞车：
#        gptqmodel/models/definitions/solar_open2.py:7
#        ImportError: cannot import name 'create_recurrent_attention_mask'
#                     from 'transformers.masking_utils'
#    gptqmodel 在 import 期会无条件加载**全部**模型定义（含 Solar-Open2），
#    而那个文件用的是 transformers >5.7 才有的 API。于是：
#        LF        要 transformers <= 5.7.0
#        gptqmodel 要 transformers >  5.7.0
#    这两者在 LF 自己的 GPTQ 导出路径上是同一条链，无法同时满足。
#    （也试过降 gptqmodel 到 6.0.3 / 5.8.0 / 5.6.12 / 5.2.0 / 4.2.5 / 2.2.0，
#      这些版本在 py3.12 上没有可用 wheel，装不上，7.3.4 原地未动。）
#
#    取舍：只把 transformers 抬到 5.14.1（quant 环境里已验证能配 gptqmodel 7.3.4），
#    其余四个仍留在 LF 区间内。这样 LF 的 check_dependencies() 只剩一条不满足，
#    用 DISABLE_VERSION_CHECK=1 跳过，再用**实际跑通导出**来验证这次跳过是安全的。
#    这个「跳过 + 实测验证」之所以可信，是因为同样的跳过在 quant 环境里 30 秒就被
#    证伪过（trl 1.x 删了 AutoModelForCausalLMWithValueHead，loader.py:28 直接
#    ImportError）——说明它确实拦得住真的 API 断裂，不是走过场。
#    Only transformers upper bound is violated; verified by a real export run.
pip install "transformers==5.14.1" "datasets==4.0.0" "peft==0.18.1" \
            "trl==0.24.0" "tyro<0.9.0" accelerate
# GPTQ 的实际执行者。gptqmodel 的 import 链里带视觉/音频分支，torchvision 与
# torchaudio 必须补上（在 quant 环境里已经踩过一次：ModuleNotFoundError: torchvision）。
pip install "gptqmodel>=2.0.0" "optimum>=1.24.0" torchvision torchaudio
# LF 本体仍用 --no-deps，避免它把上面钉好的版本重新解一遍。
pip install -e "$REPO/LLaMA-Factory" --no-deps
# LF 顶层 import 会碰到的其余模块（data/mm_plugin.py 无条件 import torchaudio 与 av）。
pip install omegaconf fire sse-starlette fastapi uvicorn av einops scipy \
            sentencepiece tiktoken

say "3/3 自检"
python - <<'PY'
import importlib
for m in ("transformers", "datasets", "peft", "trl", "gptqmodel", "llamafactory"):
    try:
        mod = importlib.import_module(m)
        print(f"  [ok] {m} = {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  [FAIL] {m}: {type(e).__name__}: {e}")
# 真正的验收：LF 的导出入口能否 import（版本检查在这一步跑）
# 这两个符号分别是上一版方案的死因和本版突破上界的原因，各验一次
try:
    from trl import AutoModelForCausalLMWithValueHead  # noqa: F401
    print("  [ok] trl.AutoModelForCausalLMWithValueHead 存在（LF loader.py:28 需要）")
except Exception as e:
    print(f"  [FAIL] trl 符号缺失: {e}")
try:
    from transformers.masking_utils import create_recurrent_attention_mask  # noqa: F401
    print("  [ok] transformers.create_recurrent_attention_mask 存在（gptqmodel 需要）")
except Exception as e:
    print(f"  [FAIL] transformers 符号缺失: {e}")
import os
os.environ["DISABLE_VERSION_CHECK"] = "1"
try:
    from llamafactory.train.tuner import export_model  # noqa: F401
    print("  [ok] llamafactory.train.tuner.export_model 可导入")
except Exception as e:
    print(f"  [FAIL] export_model: {type(e).__name__}: {e}")
PY
pip freeze > "$LOGDIR/wsl_env_lf_freeze.txt"
deactivate

say "完成"
echo "下一步（注意 DISABLE_VERSION_CHECK=1，理由见本文件 2/3 段注释）:"
echo "  source ~/venvs/lf/bin/activate"
echo "  DISABLE_VERSION_CHECK=1 python -m llamafactory.cli export Week7/configs/export_gptq_w4.yaml"
