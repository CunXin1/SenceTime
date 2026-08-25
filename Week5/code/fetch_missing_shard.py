"""补下缺失的权重分片(直连 hf-mirror.com,支持断点续传)。

## 为什么需要这个脚本

ModelScope 对 Qwen2.5-VL-7B 的 shard 1 走到了一个慢节点:实测单连接 0.6~1.4 MB/s,
中途还发生过一次 `Hash validation failed` 导致整片从头重下。
同一时刻同一条网络上,hf-mirror.com 实测 **17 MB/s**,差 12 倍。
所以缺片单独走 hf-mirror 补,已下好的分片不动。

## 为什么不用 huggingface_hub

`hf_hub_download` 在 hf-mirror 上会先请求 Xet 元数据,镜像不提供这些响应头,
直接抛 `FileMetadataError: Distant resource does not seem to be on huggingface.co`。
所以这里退到裸 HTTP 流式下载 —— 简单、可续传、实测能吃满带宽。

用法:
    .venv-vlm\\Scripts\\python.exe Week5/code/fetch_missing_shard.py Qwen2.5-VL-7B-Instruct
    .venv-vlm\\Scripts\\python.exe Week5/code/fetch_missing_shard.py gemma-4-E4B-it --dry-run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
MIRROR = "https://hf-mirror.com"
CHUNK = 8 << 20  # 8 MB

REPO = {  # 本地目录名 -> HuggingFace repo id
    "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "gemma-4-E4B-it": "google/gemma-4-E4B-it",
    # Week6 复用：ModelScope 对 3B 的 shard 2 同样掉到 178 kB/s（ETA 2h38m），
    # 与 Week5 shard 1 的情况一致，故走同一条 hf-mirror 补片路径。
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
}


def missing_shards(local: Path) -> list[str]:
    """按 index.json 列出还没落地的权重分片(只有 .incomplete 也算缺)。"""
    idx = local / "model.safetensors.index.json"
    if idx.exists():
        wanted = sorted(set(json.loads(idx.read_text(encoding="utf-8"))["weight_map"].values()))
    else:  # 单文件模型没有 index.json
        wanted = ["model.safetensors"]
    return [w for w in wanted if not (local / w).exists()]


def download(repo: str, name: str, dest: Path, max_retries: int = 30) -> None:
    """镜像会在传输中途断流(ChunkedEncodingError / IncompleteRead),
    16GB 的单文件几乎必然踩到。所以外面套重试:每次都从 .part 的当前长度续传,
    断一次只损失几秒,不会从头再来。"""
    for attempt in range(1, max_retries + 1):
        try:
            _download_once(repo, name, dest)
            return
        except Exception as exc:  # noqa: BLE001 —— 网络异常种类多,统一续传重试
            part = dest.with_suffix(dest.suffix + ".part")
            have = part.stat().st_size / 1024 ** 3 if part.exists() else 0
            if attempt == max_retries:
                raise
            print(f"  [断流 {attempt}/{max_retries}] {type(exc).__name__}: {str(exc)[:110]}\n"
                  f"  已落地 {have:.2f} GB，3 秒后从断点续传", flush=True)
            time.sleep(3)


def _download_once(repo: str, name: str, dest: Path) -> None:
    url = f"{MIRROR}/{repo}/resolve/main/{name}"
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    start_at = done   # 本次连接的起点,用于算真实速率

    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, timeout=(15, 60), headers=headers) as r:
        if done and r.status_code == 416:  # 已经下完,服务器拒绝再给
            part.rename(dest)
            return
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + done
        if done:
            print(f"  续传:已有 {done / 1024 ** 3:.2f} GB / 共 {total / 1024 ** 3:.2f} GB")
        else:
            print(f"  总大小 {total / 1024 ** 3:.2f} GB")

        t0 = last = time.perf_counter()   # 单调时钟:本机系统时钟会跳
        with part.open("ab") as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                done += len(chunk)
                if time.perf_counter() - last > 15:  # 每 15 秒打一行,别刷屏
                    el = time.perf_counter() - t0
                    # 速率只按"本次连接新收到的字节"算,不能把续传起点的存量算进去,
                    # 否则刚续上时会显示出 50 MB/s 这种假速度
                    rate = max((done - start_at) / el, 1.0)
                    print(f"  {done / 1024 ** 3:5.2f}/{total / 1024 ** 3:.2f} GB "
                          f"({done / total:5.1%})  {rate / 1024 ** 2:5.1f} MB/s  "
                          f"ETA {(total - done) / rate / 60:4.1f} min", flush=True)
                    last = time.perf_counter()

    if total and part.stat().st_size != total:
        raise RuntimeError(f"{name} 大小不符:落地 {part.stat().st_size} != 期望 {total}")
    part.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", choices=list(REPO))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    local = MODELS / args.model_dir
    todo = missing_shards(local)
    if not todo:
        print(f"[完成] {args.model_dir} 分片齐全,无需补下")
        return
    print(f"[缺失] {args.model_dir}: {todo}")
    if args.dry_run:
        return

    for name in todo:
        # 清掉 ModelScope 留下的半截文件:它的续传格式和这里的 .part 不兼容
        stale = local / f"{name}.incomplete"
        if stale.exists():
            print(f"[清理] 删除 ModelScope 残留 {stale.name} "
                  f"({stale.stat().st_size / 1024 ** 3:.2f} GB)")
            stale.unlink()

        print(f"[下载] {name} <- {MIRROR}", flush=True)
        t0 = time.perf_counter()
        download(REPO[args.model_dir], name, local / name)
        size = (local / name).stat().st_size / 1024 ** 3
        # 必须与 t0 用同一个时钟：t0 是 perf_counter()，这里若用 time.time()
        # 会得到 epoch 差值（实测打印出「29774626.5 分钟」）。
        dt = time.perf_counter() - t0
        print(f"[完成] {name}  {size:.2f} GB  {dt / 60:.1f} 分钟 "
              f"({size * 1024 / dt:.1f} MB/s)\n", flush=True)

    left = missing_shards(local)
    print(f"{'[全部齐全]' if not left else '[仍缺]'} {left or ''}")


if __name__ == "__main__":
    main()
