"""Week2：下载两个 3B 基座模型到 ./models/。

  - Qwen2.5-3B-Instruct   （HF: Qwen/Qwen2.5-3B-Instruct，非 gated）
  - Llama-3.2-3B-Instruct （HF: meta-llama/...，gated 需 token；失败回退 ModelScope）

HF 优先（走 .env 的 HF_ENDPOINT 镜像 + HF_TOKEN）。gated 模型若 403，
说明该 token 尚未在 HF 网页接受 Llama 许可 → 自动回退 ModelScope 的 LLM-Research 镜像。

用法：.venv/Scripts/python.exe Week2/code/download_models.py
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def load_env(p):
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env(ROOT / ".env")


def hf_download(repo, local_dir):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo,
        local_dir=str(local_dir),
        token=os.environ.get("HF_TOKEN"),
        # 只下推理/训练需要的文件，跳过重复的 pytorch_model.bin（用 safetensors）
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"],
    )


def ms_download(repo, local_dir):
    from modelscope import snapshot_download as ms_snap
    ms_snap(repo, local_dir=str(local_dir))


JOBS = [
    {"name": "Qwen2.5-3B-Instruct",
     "hf": "Qwen/Qwen2.5-3B-Instruct",
     "ms": "Qwen/Qwen2.5-3B-Instruct"},
    {"name": "Llama-3.2-3B-Instruct",
     "hf": "meta-llama/Llama-3.2-3B-Instruct",
     "ms": "LLM-Research/Llama-3.2-3B-Instruct"},
]


def main():
    import sys
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].lower()   # qwen / llama
    for job in JOBS:
        if only and only not in job["name"].lower():
            continue
        dest = MODELS / job["name"]
        if (dest / "config.json").exists():
            print(f"[skip] {job['name']} 已存在 -> {dest}")
            continue
        print(f"\n=== 下载 {job['name']} ===")
        try:
            print(f"  尝试 HuggingFace: {job['hf']}")
            hf_download(job["hf"], dest)
            print(f"  ✅ HF 下载完成 -> {dest}")
        except Exception as e:
            print(f"  ⚠️ HF 失败({type(e).__name__}: {str(e)[:120]})，回退 ModelScope: {job['ms']}")
            try:
                ms_download(job["ms"], dest)
                print(f"  ✅ ModelScope 下载完成 -> {dest}")
            except Exception as e2:
                print(f"  ❌ 两个源都失败：{e2}")


if __name__ == "__main__":
    main()
