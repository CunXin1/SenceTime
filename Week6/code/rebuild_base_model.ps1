# ============================================================================
#  rebuild_base_model.ps1 — Week6 Day29 前置
#  重建 Day29.1 需要的 policy 模型：models/Qwen2.5-3B-week4-dpo-merged
#
#  为什么需要这一步：
#    models/ 与 saves/ 都在 .gitignore 里，换机器后权重全部丢失，只有代码、配置
#    和数据进了 git。好在 Week3 的最优 adapter 是直接挂在**基座**上的
#    （见 Week3/configs/merge_best_qwen.yaml），不经过 Week2 的合并模型，
#    所以重建链条只有 4 步、约 40 分钟，且不必重跑任何消融实验
#    ——那些实验的结论已经写在 Week3/Week4 的交付文档里。
#
#  ★ 四步全部复用 Week3/Week4 的原始配置，一个超参都不改。
#    只有这样，重建出的模型才能和交付文档里记录的指标对得上
#    （Week3 r32 组 eval loss 1.2629；Week4 β=0.5 组 accuracies 0.974）。
#
#  ★ 用 python -m llamafactory.cli，不要用 llamafactory-cli.exe：
#    含撇号的路径（Ruibo's Desktop）会让 .exe 段错误（Week2 Day10 FAQ）。
#
#  用法（仓库根目录）:
#      powershell -ExecutionPolicy Bypass -File Week6\code\rebuild_base_model.ps1
# ============================================================================
$ErrorActionPreference = "Stop"
$ROOT = "C:\Users\Ruibo's Desktop\SenceTime_Weeks1-5"
$PY   = "$ROOT\.venv\Scripts\python.exe"
$LOG  = "$ROOT\Week6\deliverables\logs\rebuild_base_model.log"

New-Item -ItemType Directory -Force -Path (Split-Path $LOG) | Out-Null
Set-Location $ROOT

function Step($n, $name, $argv) {
    Write-Host ""
    Write-Host "=============================================================="
    Write-Host "[$n/4] $name"
    Write-Host "=============================================================="
    $t0 = Get-Date
    # ★ PowerShell 5.1 坑：对原生 exe 用 2>&1 时，stderr 的每一行都会被包成
    #   NativeCommandError 记录；在 $ErrorActionPreference="Stop" 下这会直接
    #   终止脚本——而 LLaMA-Factory 把全部 INFO 日志写在 stderr，等于一启动就死。
    #   所以在原生调用这一段临时降为 Continue，靠 $LASTEXITCODE 判断成败。
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PY @argv 2>&1 | Tee-Object -FilePath $LOG -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) {
        Write-Host "[$n/4] 失败，退出码 $code —— 链条中止" -ForegroundColor Red
        exit $code
    }
    $dt = (Get-Date) - $t0
    $msg = "[$n/4] $name 完成，耗时 {0:mm}m{0:ss}s" -f $dt
    Write-Host $msg -ForegroundColor Green
    Add-Content -Path $LOG -Value $msg -Encoding utf8
}

$T0 = Get-Date
Add-Content -Path $LOG -Value "`n===== 重建开始 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" -Encoding utf8

# ① 重跑 Week3 最优 SFT 组（r=32, lr=1e-4, ep=3）。实测 18m58s / 峰值 17.2GB。
Step 1 "Week3 SFT 重训 r32_lr1e-4_ep3" `
    @("-m", "llamafactory.cli", "train", "Week3/configs/exp/qwen_r32_lr1e-4_ep3.yaml")

# ② 合并成 week3-best-merged —— 它是 Week4 DPO 的 policy 起点。
Step 2 "合并 -> Qwen2.5-3B-week3-best-merged" `
    @("-m", "llamafactory.cli", "export", "Week3/configs/merge_best_qwen.yaml")

# ③ 重跑 Week4 最优 DPO 组（β=0.5, lr=5e-6）。实测 14m46s / 峰值 13.6GB。
#    注意不是任务书指定的 β=0.1——那组红线拒答率只有 85%，未过 90% 硬指标。
Step 3 "Week4 DPO 重训 beta0.5_lr5e-6" `
    @("-m", "llamafactory.cli", "train", "Week4/configs/exp/qwen_dpo_beta0.5_lr5e-6.yaml")

# ④ 合并成 week4-dpo-merged —— Week6 Day29 的 Agent policy。
Step 4 "合并 -> Qwen2.5-3B-week4-dpo-merged" `
    @("-m", "llamafactory.cli", "export", "Week4/configs/merge_best_dpo.yaml")

$total = (Get-Date) - $T0
$done = "===== 全部完成，总耗时 {0:hh}h{0:mm}m{0:ss}s =====" -f $total
Write-Host $done -ForegroundColor Green
Add-Content -Path $LOG -Value $done -Encoding utf8
Write-Host "Week6 Day29 的 policy 模型已就绪：models\Qwen2.5-3B-week4-dpo-merged"
