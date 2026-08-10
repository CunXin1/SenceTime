# Day24~26 剩余 GPU 任务流水线：串成一条，跑完一个立刻接下一个，GPU 不留空档。
# 每步单独记日志；某步失败不影响后续步骤（各步之间没有强依赖，除了 LoRA 训练 → lora 对比）。
#
# 用法（在 SenceTime_Week1/ 根目录）：
#   powershell -ExecutionPolicy Bypass -File Week5\code\run_rest.ps1
$ROOT = "C:\Users\Ruibo's Desktop\SenceTime_Week1"
$PY   = "$ROOT\.venv-vlm\Scripts\python.exe"
$LOG  = "$ROOT\Week5\deliverables\logs"
$env:DISABLE_VERSION_CHECK = "1"   # LF 会硬断言 transformers<=5.7.0，见 README

function Step($name, $block) {
    $t0 = [Diagnostics.Stopwatch]::StartNew()
    Write-Host "`n########## $name ##########" -ForegroundColor Cyan
    & $block
    Write-Host "########## $name 完成 $([int]$t0.Elapsed.TotalSeconds)s ##########"
}

# ---- Day24：Gemma 注意力（和 Qwen 同图同问题，才能对比两种视觉 token 策略）----
Step "Day24 gemma × 表格" {
    & $PY "$ROOT\Week5\code\attn_hook.py" --model gemma --image 01_table.png `
        --layers 5 14 20 27 `
        --question "这张表里 eval acc 最高的是哪一行？它的 eval margin 和峰值显存分别是多少？" `
        *> "$LOG\day24_gemma_table.log"
}

# ---- Day25：Gemma 幻觉探针（脚本会自动保留 Qwen 已有结果）----
Step "Day25 gemma 53 探针" {
    & $PY "$ROOT\Week5\code\hallu_eval.py" --model gemma *> "$LOG\day25_gemma.log"
}

# ---- Day26：基线必须重跑（数据已从 8 字段改版为 11 字段）----
Step "Day26 基线（11 字段版）" {
    & $PY "$ROOT\Week5\code\compare_finetune.py" --tag base *> "$LOG\day26_base.log"
}

# ---- Day26：LoRA 训练 ----
Step "Day26 LoRA 训练" {
    Push-Location $ROOT
    & $PY -m llamafactory.cli train "Week5\configs\qwen2_5vl_lora_sft.yaml" *> "$LOG\day26_train.log"
    Pop-Location
}

# ---- Day26：微调后对比（挂 adapter，不必先合并，省一次 16GB 导出）----
Step "Day26 微调后对比" {
    & $PY "$ROOT\Week5\code\compare_finetune.py" --tag lora `
        --adapter "saves\qwen2.5vl-7b-week5-lora" *> "$LOG\day26_lora.log"
}

Write-Host "`n=== 全部完成 ===" -ForegroundColor Green
