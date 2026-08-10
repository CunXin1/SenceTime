# Day26 重跑：LoRA 训练 → 微调后对比（base 结果已在 day26_raw.json 里，不重跑）。
#
# 第一次跑在 step25/69 中断，两个根因已在 configs/qwen2_5vl_lora_sft.yaml 里修掉：
#   ① image_max_pixels 589824→451584：显存贴顶后 Windows 走共享内存回退，5.4s/step → 205s/step
#   ② save_steps 100→20：总步数 69，原设置全程不落盘，一断就全丢
#
# ⚠ 不要设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。
#   实测（torch 2.6.0 + Windows）：开了之后 allocator 一上来就把 23.7/24 GB 全占满，
#   步速从 5.4 s/step 劣化到 23→31→38 s/step 且持续恶化，比不设还差。
#   expandable_segments 在 Windows 上不是支持良好的路径，这里保持默认。
#
# 用法（在 SenceTime_Week1/ 根目录）：
#   powershell -ExecutionPolicy Bypass -File Week5\code\run_day26.ps1
$ROOT = "C:\Users\Ruibo's Desktop\SenceTime_Week1"
$PY   = "$ROOT\.venv-vlm\Scripts\python.exe"
$LOG  = "$ROOT\Week5\deliverables\logs"
$env:DISABLE_VERSION_CHECK = "1"   # LF 硬断言 transformers<=5.7.0，见 README
Remove-Item Env:\PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue

function Step($name, $block) {
    $t0 = [Diagnostics.Stopwatch]::StartNew()
    Write-Host "`n########## $name ##########" -ForegroundColor Cyan
    & $block
    Write-Host "########## $name 完成 $([int]$t0.Elapsed.TotalSeconds)s ##########"
}

Step "Day26 LoRA 训练（69 步）" {
    Push-Location $ROOT
    & $PY -m llamafactory.cli train "Week5\configs\qwen2_5vl_lora_sft.yaml" *> "$LOG\day26_train.log"
    Pop-Location
}

Step "Day26 微调后对比（挂 adapter，不必先合并）" {
    & $PY "$ROOT\Week5\code\compare_finetune.py" --tag lora `
        --adapter "saves\qwen2.5vl-7b-week5-lora" *> "$LOG\day26_lora.log"
}

Write-Host "`n=== Day26 全部完成 ===" -ForegroundColor Green
