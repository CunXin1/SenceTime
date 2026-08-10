# Week5 独立环境 .venv-vlm
# 为什么要独立环境:Gemma-4 需要 transformers>=5.5.0,而主 .venv 是 4.56.2,
# 直接升级会打断 Week1~Week4 的脚本。Qwen2.5-VL 在 transformers 5.x 下同样可用,
# 所以 Week5 两个模型共用这一个新环境。
$ErrorActionPreference = "Stop"
$ROOT = "C:\Users\Ruibo's Desktop\SenceTime_Week1"
$BASE = "C:\Users\Ruibo's Desktop\AppData\Local\Programs\Python\Python312\python.exe"
$PY   = "$ROOT\.venv-vlm\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    Write-Host "[1/4] 创建 .venv-vlm"
    & $BASE -m venv "$ROOT\.venv-vlm"
}

Write-Host "[2/4] 升级 pip"
& $PY -m pip install -q -U pip setuptools wheel

Write-Host "[3/4] 安装 torch (cu124,与主环境一致)"
& $PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "[4/4] 安装 VLM 依赖"
& $PY -m pip install `
    "transformers>=5.5.0" `
    "accelerate>=1.3.0" `
    "qwen-vl-utils[decord]" `
    bitsandbytes `
    timm `
    pillow `
    matplotlib `
    pandas `
    openpyxl `
    python-docx `
    modelscope `
    "huggingface_hub[hf_transfer]" `
    datasets `
    peft

# ---------------------------------------------------------------------------
# LLaMA-Factory(Day26 微调用)。
# 冲突:LF 0.9.6.dev0 的 pyproject 把 transformers 钉在 <=5.7.0,而 Gemma-4 要 >=5.5,
# 这里装的是 5.14.1。直接 `pip install -e` 会把 transformers 降级、反过来搞坏 Gemma-4。
# 解法(已实测可用):
#   1) --no-deps 安装 LF 本体,再手工补齐它缺的依赖(跳过 transformers/accelerate/
#      datasets/peft —— 这几个已由 transformers 5.14.1 拉起,降级反而会坏)
#   2) trl 必须钉 0.24.0:trl 1.x 删掉了 AutoModelForCausalLMWithValueHead,
#      而 LF 的 model/loader.py 在模块加载时就 import 它
#   3) torchaudio 必须钉 2.6.0 匹配 torch 2.6.0,否则 _torchaudio.pyd 加载失败
#      (WinError 127)
#   4) 运行时设 DISABLE_VERSION_CHECK=1 跳过 LF 自己的 transformers 版本断言
Write-Host "[5/6] 安装 LLaMA-Factory(--no-deps)及其缺失依赖"
& $PY -m pip install -e "$ROOT\LLaMA-Factory" --no-deps -q
& $PY -m pip install -q omegaconf fire "tyro<0.9.0" einops scipy sentencepiece `
    tiktoken protobuf pydantic uvicorn fastapi sse-starlette torchdata hf-transfer
& $PY -m pip install -q "trl==0.24.0" --no-deps
& $PY -m pip install -q "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cu124

Write-Host "[6/6] 验证 LLaMA-Factory 可导入"
$env:DISABLE_VERSION_CHECK = "1"
& $PY -c "from llamafactory.train.tuner import run_exp; from llamafactory.data.template import TEMPLATES; from llamafactory.model.model_utils.visual import COMPOSITE_MODELS; print('LF OK; qwen2_vl/gemma4 模板:', 'qwen2_vl' in TEMPLATES, 'gemma4' in TEMPLATES); print('composite:', 'qwen2_5_vl' in COMPOSITE_MODELS, 'gemma4' in COMPOSITE_MODELS)"

Write-Host "`n=== 版本核对 ==="
& $PY -c "import torch,transformers,accelerate;print('torch      ',torch.__version__, torch.version.cuda);print('transformers',transformers.__version__);print('accelerate ',accelerate.__version__);print('cuda avail ',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
& $PY -c "import qwen_vl_utils, bitsandbytes; print('qwen-vl-utils OK; bitsandbytes', bitsandbytes.__version__)"
& $PY -c "from transformers import Gemma4ForConditionalGeneration, Qwen2_5_VLForConditionalGeneration; print('Gemma4 + Qwen2.5-VL 模型类均可导入')"
