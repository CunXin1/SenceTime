"""Day6：下载三个开源中文指令数据集 + 统一为 Alpaca / ShareGPT 两种标准格式。

数据集（HuggingFace 优先，走 .env 里的 HF_ENDPOINT 镜像加速）：
  1. Alpaca-GPT4-zh  —— 取 2000 条（GPT-4 生成的高质量中文指令）
  2. COIG-PC         —— 取 2000 条（BAAI 中文开放指令通用语料）
  3. ShareGPT-zh     —— 取 1000 条（多轮中英对话）

产出：
  Week2/data/raw/<name>.jsonl          原始子集（下载即存，未加工）
  Week2/data/unified/alpaca_all.json   全部样本 → Alpaca 格式 {instruction,input,output}
  Week2/data/unified/sharegpt_all.json 全部样本 → ShareGPT 格式 {conversations:[{from,value}]}
  Week2/data/unified/manifest.md/.json 原始数据归档清单（Day6 交付）

用法（在 Week1 根目录 + 已装 datasets 的 .venv 环境）：
    .venv/Scripts/python.exe Week2/code/download_data.py
"""
import os
import json
import sys
from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parents[2]          # SenceTime_Week1/
RAW_DIR = ROOT / "Week2" / "data" / "raw"
UNI_DIR = ROOT / "Week2" / "data" / "unified"
RAW_DIR.mkdir(parents=True, exist_ok=True)
UNI_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 读取 .env（HF_ENDPOINT 镜像 + HF_TOKEN）----------
def load_env(env_path: Path):
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

load_env(ROOT / ".env")
# 缓存目录放到 gitignored 的位置，避免污染 C 盘用户目录
os.environ.setdefault("HF_HOME", str(ROOT / "Week2" / "data" / "hf_cache"))

from huggingface_hub import hf_hub_download  # noqa: E402（env 设置后再 import，让镜像生效）

print(f"[env] HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', '(官方)')}")
print(f"[env] HF_HOME     = {os.environ.get('HF_HOME')}")


# ---------- 内部统一表示：把任何样本转成 turns = [(role, text), ...] ----------
# role ∈ {"human", "gpt"}。Alpaca / ShareGPT 都能由 turns 视图导出。

def clip(s):
    return (s or "").strip()


def to_alpaca(turns):
    """turns → Alpaca 单轮视图。多轮只取第一组 human/gpt（Alpaca 是单轮标准）。"""
    human = next((t for r, t in turns if r == "human"), "")
    gpt = next((t for r, t in turns if r == "gpt"), "")
    return {"instruction": clip(human), "input": "", "output": clip(gpt)}


def to_sharegpt(turns):
    """turns → ShareGPT 多轮视图 {conversations:[{from,value}]}。"""
    convs = [{"from": ("human" if r == "human" else "gpt"), "value": clip(t)}
             for r, t in turns if clip(t)]
    return {"conversations": convs}


def valid(turns):
    """至少一问一答且非空。"""
    has_h = any(r == "human" and clip(t) for r, t in turns)
    has_g = any(r == "gpt" and clip(t) for r, t in turns)
    return has_h and has_g


# ---------- 三个数据集各自的“原始行 → turns”映射 ----------
def map_alpaca_gpt4_zh(row):
    """字段：instruction / input / output（已是 Alpaca）。input 拼进 instruction。"""
    instr = clip(row.get("instruction"))
    inp = clip(row.get("input"))
    prompt = f"{instr}\n{inp}" if inp else instr
    return [("human", prompt), ("gpt", clip(row.get("output")))]


def map_coig_pc(row):
    """COIG-PC 字段同样是 instruction/input/output。"""
    instr = clip(row.get("instruction"))
    inp = clip(row.get("input"))
    prompt = f"{instr}\n{inp}" if inp else instr
    return [("human", prompt), ("gpt", clip(row.get("output")))]


def map_sharegpt_zh(row):
    """ShareGPT-zh：常见字段 'conversation'（list[{human,assistant}]）或 'conversations'。
    尽量兼容多种 schema。"""
    turns = []
    conv = row.get("conversation") or row.get("conversations") or []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        # schema A: {"human": "...", "assistant": "..."}
        if "human" in turn or "assistant" in turn:
            if clip(turn.get("human")):
                turns.append(("human", turn.get("human")))
            if clip(turn.get("assistant")):
                turns.append(("gpt", turn.get("assistant")))
        # schema B: {"from": "human"/"gpt", "value": "..."}
        elif "from" in turn and "value" in turn:
            role = "human" if turn["from"] in ("human", "user") else "gpt"
            turns.append((role, turn.get("value")))
        # schema C: {"role": "user"/"assistant", "content": "..."}
        elif "role" in turn and "content" in turn:
            role = "human" if turn["role"] in ("human", "user") else "gpt"
            turns.append((role, turn.get("content")))
    return turns


# ---------- 数据集清单（repo、split、取数、映射函数）----------
# 说明：COIG-PC / ShareGPT-zh 的确切 repo/字段跑一次就能确认，若失败按 fallback 调整。
DATASETS = [
    {
        "name": "alpaca_gpt4_zh",
        "hf_repo": "llamafactory/alpaca_gpt4_zh",
        "file": "alpaca_gpt4_data_zh.json",           # 单个 json 文件
        "kind": "json",
        "n": 2000,
        "mapper": map_alpaca_gpt4_zh,
        "license": "Apache-2.0 (data: GPT-4 generated)",
        "note": "GPT-4 生成的中文指令-回答对，LLaMA-Factory 官方镜像",
    },
    {
        "name": "coig_pc",
        "hf_repo": "BAAI/COIG-PC-Lite",
        "file": "data/train-00000-of-00001-175fcfc36f67f974.parquet",  # train 分片 parquet
        "kind": "parquet",
        "n": 2000,
        "mapper": map_coig_pc,
        "license": "见 BAAI/COIG-PC 数据卡（各子任务不同）",
        "note": "BAAI 中文开放指令通用语料（Lite 精选子集）",
    },
    {
        "name": "sharegpt_zh",
        "hf_repo": "shareAI/ShareGPT-Chinese-English-90k",
        "file": "shareGPT/computer_zh_26k.jsonl",     # 中文多轮对话 jsonl
        "kind": "jsonl",
        "n": 1000,
        "mapper": map_sharegpt_zh,
        "license": "Apache-2.0",
        "note": "多轮中文对话（ShareGPT 90k 的中文子集）",
    },
]


def take_n(cfg):
    """下载指定文件到本地后解析，取前 n 条。比 streaming 自动解析更稳。"""
    path = hf_hub_download(repo_id=cfg["hf_repo"], filename=cfg["file"],
                           repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    kind, n = cfg["kind"], cfg["n"]
    rows = []
    if kind == "json":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = data[:n]
    elif kind == "jsonl":
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif kind == "parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        rows = df.head(n).to_dict("records")
    return rows


def main():
    alpaca_all, sharegpt_all = [], []
    manifest = []

    for cfg in DATASETS:
        name = cfg["name"]
        print(f"\n=== 下载 {name}  <- {cfg['hf_repo']} (取 {cfg['n']} 条) ===")
        try:
            rows = take_n(cfg)
        except Exception as e:
            print(f"[ERROR] {name} 下载失败：{e}")
            manifest.append({"name": name, "repo": cfg["hf_repo"], "status": f"FAILED: {e}",
                             "requested": cfg["n"], "downloaded": 0})
            continue

        # 存原始子集
        raw_path = RAW_DIR / f"{name}.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"  原始 {len(rows)} 条 -> {raw_path.relative_to(ROOT)}")
        if rows:
            print(f"  首行字段: {list(rows[0].keys())}")

        # 转统一格式
        kept = 0
        for r in rows:
            turns = cfg["mapper"](r)
            if not valid(turns):
                continue
            alpaca_all.append(to_alpaca(turns))
            sharegpt_all.append(to_sharegpt(turns))
            kept += 1
        print(f"  有效转换 {kept} 条")

        manifest.append({
            "name": name, "repo": cfg["hf_repo"], "status": "OK",
            "requested": cfg["n"], "downloaded": len(rows), "valid": kept,
            "license": cfg["license"], "note": cfg["note"],
        })

    # 写统一格式文件
    (UNI_DIR / "alpaca_all.json").write_text(
        json.dumps(alpaca_all, ensure_ascii=False, indent=2), encoding="utf-8")
    (UNI_DIR / "sharegpt_all.json").write_text(
        json.dumps(sharegpt_all, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n统一格式: Alpaca {len(alpaca_all)} 条, ShareGPT {len(sharegpt_all)} 条")

    # 写归档清单（json + md）
    (UNI_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest_md(manifest, len(alpaca_all))
    print(f"归档清单: {(UNI_DIR / 'manifest.md').relative_to(ROOT)}")


def write_manifest_md(manifest, total):
    lines = ["# Day6 原始数据归档清单\n",
             f"> 数据源：HuggingFace（经 `{os.environ.get('HF_ENDPOINT','官方')}` 镜像）\n",
             "| 数据集 | HF 仓库 | 请求 | 下载 | 有效 | 许可 | 说明 |",
             "|---|---|---|---|---|---|---|"]
    for m in manifest:
        lines.append(
            f"| {m['name']} | `{m['repo']}` | {m.get('requested','-')} | "
            f"{m.get('downloaded','-')} | {m.get('valid','-')} | "
            f"{m.get('license','-')} | {m.get('note', m.get('status',''))} |")
    lines += ["",
              f"**合计有效样本：{total} 条**（已统一为 Alpaca 与 ShareGPT 两种格式）",
              "",
              "## 产出文件",
              "- `Week2/data/raw/*.jsonl` —— 各数据集原始子集（未加工，gitignored）",
              "- `Week2/data/unified/alpaca_all.json` —— 统一 Alpaca 格式 `{instruction,input,output}`",
              "- `Week2/data/unified/sharegpt_all.json` —— 统一 ShareGPT 格式 `{conversations:[{from,value}]}`",
              "",
              "> 下一步（Day7）：`clean_pipeline.py` 对 `alpaca_all.json` 去噪、去重、截断，产出 ≥1500 条清洗集。"]
    (UNI_DIR / "manifest.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
