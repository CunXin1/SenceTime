"""
app.py — Week7 Day37 / Day38
Gradio 前端：对接 vLLM 的 OpenAI 兼容接口，流式对话 + 图片上传。
Gradio front-end for the vLLM OpenAI-compatible server (streaming + images).

设计目标（任务书原话）：让一个不懂深度学习的同学也能上传图片或输入文字与模型对话。
下面每个取舍都是朝"对方不需要知道背后有什么"这个方向做的。

★ 为什么后端切换做成一个下拉框，而不是起两个应用
    24GB 显存塞不下 3B 文本模型 + 7B VLM 同时常驻（fp16 的 VLM 单独就要约 16GB）。
    所以两个服务是**互斥槽位**：同一时刻只起一个，靠端口区分（8000 文本 / 8001 多模态）。
    UI 侧把这件事藏起来——用户只看到"聊天模型 / 看图模型"两个选项，切换后自动重连，
    连不上时给出**能直接照抄的启动命令**，而不是甩一个 Connection refused。
    Two mutually exclusive server slots (VRAM-bound); the UI hides the swap.

★ 为什么自己维护一份 api_history，而不是把 Chatbot 的内容回传给模型
    Gradio 的 Chatbot 为了渲染，会把图片存成 {"path": ...} 这类前端结构。
    直接拿去喂 API 需要反向解析，Gradio 一升版就碎。
    这里用 gr.State 单独存一份**纯 OpenAI 格式**的历史：渲染归渲染、协议归协议，
    两边只在提交时同步一次。多写二十行，换掉一整类"升级后聊天记录错乱"的 bug。
    Keep a separate OpenAI-format history in gr.State; never reverse-parse the UI.

★ 为什么图片走 base64 data URI 而不是传路径
    vLLM 在 WSL2 里，Gradio 在 Windows 上，两者**不共享文件系统视图**：
    Windows 的 C:\\Users\\... 到了 WSL 是 /mnt/c/Users/...，直接传路径必然找不到文件。
    base64 内联进请求体，跨进程、跨文件系统、跨机器都成立，代价是请求体膨胀约 33%。
    本地单用户场景这点开销无所谓。
    The server lives in WSL with a different filesystem view; inline the bytes.

★ 流式为什么必须是 yield 而不是等全部返回
    3B 模型生成 256 token 约 1-2 秒，7B VLM 看图更久。用户盯着转圈会觉得"卡住了"，
    而首 token 0.03 秒就吐出来，体感是"秒回"。实测 TTFT=0.033s（client_demo.py 的输出）
    ——这个数字才是体验的决定因素，总吞吐决定的只是"读起来顺不顺"。
    TTFT drives perceived latency; that's why streaming is non-negotiable here.

用法 / Usage（Windows 侧的主 .venv；vLLM 在 WSL 里跑）:
    .venv\\Scripts\\python.exe Week7/code/app.py
    .venv\\Scripts\\python.exe Week7/code/app.py --share      # 生成公网临时链接
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from pathlib import Path

# ★ 这三行必须在 import gradio 之前，且顺序不能动（2026-08-21 实测踩坑之二）
#   Gradio 启动时会**请求它自己**的 /gradio_api/startup-events 做自检。本机系统代理
#   （HKCU\...\Internet Settings → ProxyServer=127.0.0.1:7897）会把这个 localhost
#   请求也劫走，于是 launch() 抛：
#       Couldn't start the app because '.../startup-events' failed (code 502).
#   为什么设 NO_PROXY 就能根治：urllib.request.getproxies() 的实现是
#       getproxies_environment() or getproxies_registry()
#   ——只要环境变量里有任何代理相关的键（含 no_proxy），前者返回非空，
#   **注册表那一路就整个不读了**。而我们只填 no_proxy 不填 http_proxy，
#   于是结果就是"什么代理都不走"，正是本地服务想要的。
#   放在 import gradio 之前是因为要赶在它构造内部 httpx client 之前生效。
#   (make_client() 里的 trust_env=False 是同一个坑的另一半：那管的是我们自己发出的
#    请求，这里管的是 Gradio 内部发出的请求，两处都要堵。)
#   Gradio self-requests localhost at startup; the system proxy hijacks it (502).
#   Setting no_proxy makes urllib skip the registry lookup entirely.
for _k in ("NO_PROXY", "no_proxy"):
    os.environ[_k] = "localhost,127.0.0.1,::1"

import gradio as gr  # noqa: E402
import httpx  # noqa: E402
from openai import OpenAI  # noqa: E402

# ---------------------------------------------------------------------------
# 后端槽位。两个服务互斥，同一时刻只起一个（显存不够同时常驻）。
# port 与 Week7/code/serve_vllm.sh 的约定一致。
BACKENDS = {
    "聊天模型（Qwen2.5-3B · AWQ 4-bit）": {
        "port": 8000, "model": "qwen3b", "multimodal": False,
        "serve_cmd": "bash Week7/code/serve_vllm.sh awq",
    },
    "看图模型（Qwen2.5-VL-7B）": {
        "port": 8001, "model": "qwen-vl", "multimodal": True,
        "serve_cmd": "bash Week7/code/serve_vllm.sh vl 8001",
    },
}
DEFAULT_BACKEND = next(iter(BACKENDS))

SYSTEM_PROMPT = "你是一个乐于助人的中文助手，回答简洁、准确。"


def make_client(port: int) -> OpenAI:
    """构造指向本地 vLLM 的 OpenAI 客户端。

    ★ trust_env=False 是必须的，否则本机开了代理就 502（2026-08-21 实测踩坑）
        症状很有迷惑性：curl 打 127.0.0.1:8000 返回 200，端口转发一切正常，
        但 openai SDK 一调就 `InternalServerError: Error code: 502`。
        链条是这样的：
          1) openai SDK 底层用 httpx；httpx 在 trust_env=True（默认）时
             调 urllib.request.getproxies() 取代理配置；
          2) 该函数在 Windows 上会去读**注册表**里的系统代理
             （本机 HKCU\\...\\Internet Settings → ProxyServer=127.0.0.1:7897）；
          3) 注册表里其实有 ProxyOverride 白名单，包含 localhost 和 127.*，
             但 getproxies() **不返回 'no' 键**，httpx 也不实现 ProxyOverride 的
             绕过逻辑——于是它把发往 127.0.0.1:8000 的请求塞进了代理，
             代理转不了 localhost，回一个 502。
        所以这不是"服务没起来"，而是"请求绕了一圈没到服务"。
        关掉 trust_env 让 httpx 直连，比要求用户去改系统代理设置健壮得多——
        毕竟目标是"不懂深度学习的同学也能用"。
        httpx reads the Windows registry proxy but ignores ProxyOverride, so
        localhost requests get proxied and 502. Disable env-based proxying.
    """
    # api_key 随便填：vLLM 默认不校验，但 openai SDK 强制要求非空（沿用 client_demo.py）
    return OpenAI(
        base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY",
        http_client=httpx.Client(trust_env=False, timeout=600.0),
    )


def image_to_data_uri(path: str | Path) -> str:
    """读图 -> base64 data URI。理由见文件头第三条。"""
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def check_backend(name: str) -> str:
    """探活。连不上时给出能直接照做的下一步，而不是把异常甩给用户。"""
    cfg = BACKENDS[name]
    try:
        ids = [m.id for m in make_client(cfg["port"]).models.list().data]
    except Exception as exc:  # noqa: BLE001
        return (f"🔴 **{name}** 未连接（127.0.0.1:{cfg['port']}）\n\n"
                f"请在 WSL 里执行：\n```\n{cfg['serve_cmd']}\n```\n"
                f"<sub>{type(exc).__name__}</sub>")
    return f"🟢 **{name}** 已就绪 · 端口 {cfg['port']} · 模型 `{', '.join(ids)}`"


def build_user_content(text: str, files: list, multimodal: bool):
    """把 Gradio 的 {text, files} 转成 OpenAI 的 content。

    纯文本时返回**字符串**而不是 [{"type": "text"}] 数组：文本模型的服务端不接受
    多模态 content 数组。同一个前端要能打两种后端，这里就是分岔点。
    """
    if not files or not multimodal:
        return text
    parts = []
    for f in files:
        parts.append({"type": "image_url", "image_url": {"url": image_to_data_uri(f)}})
    if text:
        parts.append({"type": "text", "text": text})
    return parts


def on_submit(message: dict, chat_history: list):
    """用户按下发送：先把消息渲染出去、清空输入框，真正的生成交给 bot_stream。"""
    message = message or {}
    text = message.get("text") or ""
    files = message.get("files") or []
    if not text.strip() and not files:
        return gr.update(), chat_history

    for f in files:
        chat_history = chat_history + [{"role": "user", "content": {"path": f}}]
    if text.strip():
        chat_history = chat_history + [{"role": "user", "content": text}]
    return gr.update(value=None), chat_history


def bot_stream(chat_history: list, api_history: list, backend: str,
               temperature: float, top_p: float, max_tokens: float,
               raw_message: dict):
    """调模型并流式回填。yield 的每一帧都会立刻推到浏览器。"""
    cfg = BACKENDS[backend]
    raw_message = raw_message or {}
    text = raw_message.get("text") or ""
    files = raw_message.get("files") or []

    if files and not cfg["multimodal"]:
        chat_history = chat_history + [{
            "role": "assistant",
            "content": "⚠️ 当前是**聊天模型**，看不了图片。请在左侧切换到「看图模型」，"
                       "并确认对应服务已启动。",
        }]
        yield chat_history, api_history
        return

    content = build_user_content(text, files, cfg["multimodal"])
    api_history = api_history + [{"role": "user", "content": content}]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + api_history

    chat_history = chat_history + [{"role": "assistant", "content": ""}]
    acc = ""
    try:
        stream = make_client(cfg["port"]).chat.completions.create(
            model=cfg["model"], messages=messages,
            temperature=float(temperature), top_p=float(top_p),
            max_tokens=int(max_tokens), stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            acc += delta
            chat_history[-1]["content"] = acc
            yield chat_history, api_history
    except Exception as exc:  # noqa: BLE001
        chat_history[-1]["content"] = (
            f"🔴 调用失败：{type(exc).__name__}: {exc}\n\n"
            f"检查服务是否在跑：`{cfg['serve_cmd']}`")
        yield chat_history, api_history
        return

    api_history = api_history + [{"role": "assistant", "content": acc}]
    yield chat_history, api_history


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Week7 · 本地大模型对话", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 本地大模型对话\n"
            "第 4 周 DPO 微调 → 第 7 周 AWQ 4-bit 量化 → vLLM 服务化。"
            "**权重显存降低 66%，首 token 延迟 0.03 秒。**"
        )
        api_history = gr.State([])
        last_message = gr.State({})

        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                backend = gr.Dropdown(
                    choices=list(BACKENDS), value=DEFAULT_BACKEND,
                    label="后端模型",
                    info="两个服务互斥：显存不够同时常驻，切换前先起对应服务",
                )
                status = gr.Markdown(check_backend(DEFAULT_BACKEND))
                refresh = gr.Button("重新检测连接", size="sm")

                gr.Markdown("### 生成参数")
                # ★ 取值范围按"非专业用户也调不出灾难"来定：
                #   temperature 上限 1.5 而非 2.0（再高 3B 模型开始胡言乱语），
                #   top_p 下限 0.1（更低会退化成近似贪心，失去调节的意义）。
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05,
                                        label="Temperature",
                                        info="越高越发散；0 = 每次回答都一样")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p",
                                  info="只从累计概率前 p 的词里挑，越低越保守")
                max_tokens = gr.Slider(64, 2048, value=512, step=64,
                                       label="最大生成长度")
                clear = gr.Button("清空对话", variant="secondary")

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    type="messages", height=520, show_copy_button=True,
                    placeholder="### 试试问我\n"
                                "- 用三句话解释什么是模型量化\n"
                                "- 鸡兔同笼：35 个头、94 只脚，鸡兔各几只？\n"
                                "- （切到看图模型后）上传一张图片问我看到了什么",
                )
                textbox = gr.MultimodalTextbox(
                    file_types=["image"], file_count="multiple",
                    placeholder="输入消息，或点回形针上传图片…",
                    show_label=False,
                )

        refresh.click(check_backend, [backend], [status])
        backend.change(check_backend, [backend], [status])

        # 链式：先把原始消息存进 State（因为下一步就要清空输入框），
        # 再渲染用户气泡，最后交给流式生成。
        textbox.submit(
            lambda m: m, [textbox], [last_message],
        ).then(
            on_submit, [textbox, chatbot], [textbox, chatbot],
        ).then(
            bot_stream,
            [chatbot, api_history, backend, temperature, top_p, max_tokens, last_message],
            [chatbot, api_history],
        )

        clear.click(lambda: ([], [], {}), None, [chatbot, api_history, last_message])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="生成公网临时链接")
    args = ap.parse_args()
    build_ui().queue().launch(server_name=args.host, server_port=args.port,
                              share=args.share, inbrowser=False)


if __name__ == "__main__":
    main()
