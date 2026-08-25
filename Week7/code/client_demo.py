"""
client_demo.py — Week7 Day36.2
用官方 openai SDK 打 vLLM 的 /v1/chat/completions，验证服务可用（验收标准❷）。
Python client for the vLLM OpenAI-compatible endpoint.

★ 为什么用 openai SDK 而不是手写 requests
    vLLM 提供的就是 OpenAI 兼容接口，用官方 SDK 打通，等于同时验证了"任何 OpenAI
    生态的工具（LangChain / OpenWebUI / 自研前端）都能零改造接上"——这才是
    "OpenAI 兼容"这四个字的价值。Day37 的 Gradio 前端复用同一个 client 构造方式。

★ api_key 为什么随便填
    vLLM 默认不校验 key，但 openai SDK 强制要求非空，所以填 "EMPTY"（vLLM 文档的
    惯例值）。若要开鉴权，服务端加 --api-key 后这里同步改。

用法 / Usage（WSL 的 vllm 环境；也可在 Windows 侧跑，靠 WSL2 localhost 转发）:
    python Week7/code/client_demo.py                 # 非流式 + 流式各跑一遍
    python Week7/code/client_demo.py --stream-only
    python Week7/code/client_demo.py --port 8000 --model qwen3b
"""

from __future__ import annotations

import argparse
import time

import httpx
from openai import OpenAI

QUESTION = "用两句话说明 AWQ 和 GPTQ 的核心区别。"


def make_client(base: str, port: int) -> OpenAI:
    # ★ trust_env=False：从 Windows 侧调用时，httpx 会读注册表里的系统代理却不认
    #   ProxyOverride 白名单，把 127.0.0.1 的请求也塞进代理 → 502。
    #   在 WSL 里跑碰不到（WSL 内没有代理配置），但 Day37 的 Gradio 跑在 Windows 上，
    #   同一个坑必踩。详细推理见 Week7/code/app.py 的 make_client()。
    return OpenAI(base_url=f"{base}:{port}/v1", api_key="EMPTY",
                  http_client=httpx.Client(trust_env=False, timeout=600.0))


def show_models(cli: OpenAI) -> None:
    print("=== /v1/models ===")
    for m in cli.models.list().data:
        print(f"  id={m.id}  owned_by={m.owned_by}")


def chat_once(cli: OpenAI, model: str, temperature: float, top_p: float) -> None:
    print("\n=== /v1/chat/completions (非流式) ===")
    t0 = time.time()
    r = cli.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": QUESTION}],
        temperature=temperature, top_p=top_p, max_tokens=256,
    )
    dt = time.time() - t0
    u = r.usage
    print(r.choices[0].message.content)
    print(f"\n[usage] prompt={u.prompt_tokens} completion={u.completion_tokens} "
          f"耗时={dt:.2f}s 平均={u.completion_tokens / dt:.1f} tok/s")


def chat_stream(cli: OpenAI, model: str, temperature: float, top_p: float) -> None:
    """流式：Day37 的 Gradio 就是把这里的 yield 接到 ChatInterface 上。

    首 token 延迟(TTFT)单独计时——它决定用户"感觉快不快"，而总 tokens/s 决定
    "读起来顺不顺"。两个指标是分开的，UI 体验主要吃前者。
    TTFT drives perceived latency; total tok/s drives reading smoothness.
    """
    print("\n=== /v1/chat/completions (stream=True) ===")
    t0 = time.time()
    ttft = None
    n = 0
    stream = cli.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": QUESTION}],
        temperature=temperature, top_p=top_p, max_tokens=256,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if not delta:
            continue
        if ttft is None:
            ttft = time.time() - t0
        n += 1
        print(delta, end="", flush=True)
    dt = time.time() - t0
    print(f"\n\n[stream] TTFT={ttft:.3f}s 总耗时={dt:.2f}s chunk数={n} "
          f"≈{n / dt:.1f} chunk/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen3b")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--stream-only", action="store_true")
    args = ap.parse_args()

    cli = make_client(args.base, args.port)
    try:
        show_models(cli)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"[FAIL] 连不上 {args.base}:{args.port} ({type(exc).__name__}: {exc})\n"
            "  · 服务起了吗：bash Week7/code/serve_vllm.sh awq\n"
            "  · 若从 Windows 侧调用 WSL 里的服务，确认 serve 时用了 --host 0.0.0.0"
        ) from exc

    if not args.stream_only:
        chat_once(cli, args.model, args.temperature, args.top_p)
    chat_stream(cli, args.model, args.temperature, args.top_p)


if __name__ == "__main__":
    main()
