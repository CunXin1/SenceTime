# -*- coding: utf-8 -*-
"""
Week8 Day42 · 学生模型下载脚本
Fetch the student base model (Qwen2.5-0.5B-Instruct) from hf-mirror via plain HTTP.

================================ 为什么需要这个脚本 ================================
Day42 的知识蒸馏需要一个"学生"基座 Qwen2.5-0.5B-Instruct。本地 models/ 下只有 3B
系列（教师 + 其 AWQ/GPTQ 量化版），0.5B 从未下载过，必须先拉下来。

★ 取舍 1：走 hf-mirror，不走 ModelScope
   Week5/Week6 两次实测：ModelScope 对 Qwen 权重会掉到 ~178 kB/s（1GB 要一个多小时），
   同一条网络上 hf-mirror 有 17~35 MB/s。所以固定用 hf-mirror。
   (Mirror chosen on measured throughput, not preference.)

★ 取舍 2：裸 HTTP 流式下载，不用 huggingface_hub —— 这是本脚本存在的真正理由
   本机实测（2026-08-25）：即便 HF_ENDPOINT=https://hf-mirror.com 已生效
   （constants.ENDPOINT 确认是 hf-mirror），`hf_hub_download` 仍然抛：
       FileMetadataError: Distant resource does not seem to be on huggingface.co
   原因是 hub 的 HEAD 探测依赖 Xet / X-Linked-ETag 一族响应头，镜像的 308→307→200
   重定向链上这些头不完整，hub 判定"这不是 HF"直接放弃，再退化成 LocalEntryNotFound。
   Week5/code/fetch_missing_shard.py 的 docstring 里已经记录过同一个坑，此处沿用同一解法：
   requests.get(stream=True) 裸下 + Range 断点续传。
   (huggingface_hub's metadata HEAD probe is incompatible with the mirror; use raw HTTP.)

★ 取舍 3：下到 models/ 下的普通目录，不进 HF cache
   HF cache 是 blob + symlink 结构，Windows 上非管理员建符号链接会退化成复制，
   等于同一份权重占两份磁盘。C 盘只剩 35GB，直接落普通目录最省。

用法：
    .venv/Scripts/python.exe Week8/scripts/download_student.py
    .venv/Scripts/python.exe Week8/scripts/download_student.py --check   # 只校验不下载
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
MIRROR = "https://hf-mirror.com"
REPO = "Qwen/Qwen2.5-0.5B-Instruct"
DEST = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
CHUNK = 8 << 20  # 8 MB —— 与 Week5 脚本保持一致，实测能吃满带宽

# 只列训练/推理真正需要的文件。README/LICENSE/.gitattributes 一律不下。
FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def _download_once(name: str, dest: Path) -> None:
    """单次尝试：从 .part 的当前长度续传。"""
    url = f"{MIRROR}/{REPO}/resolve/main/{name}"
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}

    with requests.get(url, stream=True, timeout=(15, 60), headers=headers) as r:
        if done and r.status_code == 416:  # 服务器说"你已经下完了"
            part.rename(dest)
            return
        r.raise_for_status()
        total = done + int(r.headers.get("Content-Length", 0))
        t0, last = time.time(), done
        with open(part, "ab") as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                dt = time.time() - t0
                if dt > 2:
                    mbps = (done - last) / dt / 1024**2
                    pct = 100 * done / total if total else 0
                    print(f"    {name}: {done/1024**2:7.1f} MiB ({pct:5.1f}%) {mbps:5.1f} MB/s",
                          flush=True)
                    t0, last = time.time(), done
    part.rename(dest)


def download(name: str, dest: Path, max_retries: int = 20) -> None:
    """★ 镜像会在传输中途断流（ChunkedEncodingError / IncompleteRead），
    大文件几乎必踩。外层重试 + Range 续传，断一次只损失几秒。"""
    for attempt in range(1, max_retries + 1):
        try:
            _download_once(name, dest)
            return
        except Exception as exc:  # noqa: BLE001 —— 网络异常种类多，统一按"续传重试"处理
            if attempt == max_retries:
                raise
            part = dest.with_suffix(dest.suffix + ".part")
            have = part.stat().st_size / 1024**2 if part.exists() else 0
            print(f"  [断流 {attempt}/{max_retries}] {type(exc).__name__}: {str(exc)[:100]}\n"
                  f"  已落地 {have:.1f} MiB，3 秒后续传", flush=True)
            time.sleep(3)


def verify() -> bool:
    """自检：必需文件都在，且 config.json 能解析出 vocab_size。"""
    missing = [f for f in FILES if not (DEST / f).exists()]
    if missing:
        print(f"[verify] 缺文件: {missing}")
        return False
    import json
    cfg = json.loads((DEST / "config.json").read_text(encoding="utf-8"))
    size_mb = (DEST / "model.safetensors").stat().st_size / 1024**2
    print(f"[verify] OK  vocab_size={cfg['vocab_size']}  hidden={cfg['hidden_size']}  "
          f"layers={cfg['num_hidden_layers']}  weights={size_mb:.1f} MiB")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验已下载的文件，不发起下载")
    args = ap.parse_args()

    if args.check:
        return 0 if verify() else 1

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"[download_student] {REPO} -> {DEST}")
    for name in FILES:
        target = DEST / name
        if target.exists():
            print(f"  [skip] {name} 已存在")
            continue
        print(f"  [get ] {name}", flush=True)
        download(name, target)
    return 0 if verify() else 1


if __name__ == "__main__":
    sys.exit(main())
