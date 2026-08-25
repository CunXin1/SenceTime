#!/usr/bin/env bash
# ============================================================================
#  run_pipeline.sh — 仓库根入口 / Repository-root entry point
#
#  这是一层**薄壳**，真正的编排在 Week8/scripts/run_pipeline.sh 里。
#  This is a thin shim; the real orchestration lives in Week8/scripts/.
#
#  ★ 为什么要在根目录再放一个入口
#    README 的「快速开始」希望别人 clone 下来第一条命令就能跑，
#    而不是先去读目录结构、找到 Week8/scripts/ 才知道从哪起手。
#    同时它负责一件薄壳该干的事：**把工作目录固定到仓库根**——
#    项目里所有脚本都用 `Path(__file__).resolve().parents[2]` 或 `$REPO` 定位，
#    但 LLaMA-Factory 的 dataset_dir 等少数配置是相对工作目录解析的，
#    在别处 cd 进来直接跑会找不到数据集。
#    Pins the CWD to the repo root: a few LLaMA-Factory paths resolve
#    relative to the working directory, not to the script location.
#
#  用法 / Usage:
#      bash run_pipeline.sh --help
#      bash run_pipeline.sh --quick                  # 冒烟：小步数跑完整链路
#      bash run_pipeline.sh --skip-train             # 只做数据 + 评估
#      bash run_pipeline.sh --only eval
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL="$REPO/Week8/scripts/run_pipeline.sh"

if [ ! -f "$REAL" ]; then
    echo "[FAIL] 找不到主控脚本：$REAL"
    echo "       仓库结构可能不完整；见 README.md 的「目录结构」一节。"
    exit 1
fi

cd "$REPO"                      # ← 薄壳存在的理由，见上方注释
exec bash "$REAL" "$@"
