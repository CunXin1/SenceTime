"""Week5 VLM 模型下载(ModelScope 国内源,失败自动回退 HuggingFace 镜像)。

两个模型:
  1. Qwen/Qwen2.5-VL-7B-Instruct  —— 中国 VLM,原生动态分辨率 ViT,视觉 token 数随图变
  2. google/gemma-4-E4B-it        —— 美国 VLM,~150M vision encoder(2D 学习位置 + 多维 RoPE),
                                     soft token 预算固定可选 70/140/280/560/1120

用法(主 .venv 即可,只依赖 modelscope):
    .venv\\Scripts\\python.exe Week5/code/download_vlm.py
    .venv\\Scripts\\python.exe Week5/code/download_vlm.py --only qwen
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

# key -> (modelscope_id, huggingface_id, 本地目录名, 预计体积GB)
SPECS = {
    "qwen": ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct",
             "Qwen2.5-VL-7B-Instruct", 16.6),
    "gemma": ("google/gemma-4-E4B-it", "google/gemma-4-E4B-it",
              "gemma-4-E4B-it", 16.0),
}


def human_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024 ** 3


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024 ** 3


def download_one(key: str) -> Path:
    ms_id, hf_id, local_name, est = SPECS[key]
    target = MODELS_DIR / local_name
    print(f"\n{'=' * 70}\n[{key}] {ms_id} -> {target}  (预计 {est} GB)\n{'=' * 70}", flush=True)

    if target.exists() and (target / "config.json").exists():
        got = human_gb(target)
        if got > est * 0.9:
            print(f"[跳过] 已存在且体积正常 ({got:.1f} GB)")
            return target
        print(f"[续传] 已存在但只有 {got:.1f} GB,继续下载")

    if free_gb(MODELS_DIR) < est + 5:
        sys.exit(f"[中止] 磁盘剩余 {free_gb(MODELS_DIR):.1f} GB,不足以放下 {est} GB 模型")

    t0 = time.time()
    try:
        from modelscope import snapshot_download as ms_download
        ms_download(ms_id, local_dir=str(target))
    except Exception as exc:  # noqa: BLE001 —— 国内网络波动/模型未同步都可能触发
        print(f"[ModelScope 失败] {type(exc).__name__}: {exc}\n[回退] 改用 HuggingFace 镜像 hf-mirror.com", flush=True)
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from huggingface_hub import snapshot_download as hf_download
        hf_download(hf_id, local_dir=str(target), max_workers=4)

    print(f"[完成] {local_name}: {human_gb(target):.1f} GB, 耗时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    return target


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(SPECS), help="只下载其中一个")
    args = ap.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    keys = [args.only] if args.only else list(SPECS)
    print(f"磁盘剩余 {free_gb(MODELS_DIR):.1f} GB,准备下载: {keys}")
    for k in keys:
        download_one(k)

    print("\n=== 全部完成 ===")
    for k in keys:
        p = MODELS_DIR / SPECS[k][2]
        print(f"  {SPECS[k][2]:<32} {human_gb(p):>6.1f} GB  {p}")
