"""从 HuggingFace 下载模型到 ./models/ 目录。

用法(在 Week1/ 根目录 + llm_exp 环境运行):
    conda activate llm_exp && cd ~/SenceTime/Week1
    python code/download_models.py            下载全部
"""
import os
import sys
from huggingface_hub import snapshot_download

# 从 .env 读取 HF_TOKEN 并写入环境变量;
# snapshot_download 会自动使用该变量,从而下载受限(gated)模型。
for line in open(".env"):
    if line.startswith("HF_TOKEN="):
        os.environ["HF_TOKEN"] = line.strip().split("=", 1)[1]

# 命令行 key -> HuggingFace 仓库 id
MODELS = {
    "qwen2": "Qwen/Qwen2.5-1.5B-Instruct",       # 作业主角
    "qwen3":  "Qwen/Qwen3-4B",                    # 代际对比
    "llama":  "meta-llama/Llama-3.2-3B-Instruct",
    "gemma":  "google/gemma-4-E2B-it",
}

# 只下载权重、配置和分词器文件,跳过其它冗余格式
# 注意:*.jinja 必须包含,否则 Gemma 等模型的 chat_template.jinja 会缺失,推理时报错
ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "tokenizer*", "*.jinja"]


def download(key):
    repo_id = MODELS[key]
    local_dir = f"./models/{repo_id.split('/')[-1]}"
    print(f"开始下载 {repo_id}")
    # snapshot_download 支持断点续传,已存在的文件会跳过
    snapshot_download(repo_id, local_dir=local_dir, allow_patterns=ALLOW_PATTERNS)
    print(f"完成 {repo_id} -> {local_dir}")


if __name__ == "__main__":
    # 无参数时下载全部,否则只下载命令行指定的模型
    keys = sys.argv[1:] or list(MODELS)
    for key in keys:
        download(key)
