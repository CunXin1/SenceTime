# 第2周 FAQ：报错与解决方案（4090 / Windows 11 / Python 3.12 venv）

> 本周在全新 4090 Windows 机器上从零搭环境，踩坑集中在**依赖版本**和 **Windows 特有行为**。
> 每条含：现象 → 根因 → 解决。定位方法可复用。

---

## 一、原生段错误（Segmentation fault，exit 139）——最难查

### 1.1 `transformers 5.7.0` 导入 Trainer 崩溃
- **现象**：`llamafactory-cli train` 秒退、exit 139、**日志 0 行**（崩在任何输出前）。
- **定位**：`faulthandler.enable()` + 逐个 import，发现 `from transformers.trainer import Trainer` 崩。
- **根因**：LLaMA-Factory 0.9.6 允许 `transformers<=5.7.0`，pip 装了最高的 **5.7.0**（5.x 大改版），在 Win/py3.12 上不稳。
- **解决**：降级到 4.x —— `pip install "transformers>=4.55,<4.57"`（装到 4.56.2）。

### 1.2 `pyarrow 24.0.0` 在 torch 之后加载崩溃
- **现象**：降完 transformers 后，训练仍 exit 139、0 行日志。
- **定位**：`faulthandler` 抓到崩在 `pyarrow/__init__.py:71`；最小复现 `import torch; from transformers.trainer import Trainer`（Trainer→datasets→pandas→pyarrow）。
- **根因**：**pyarrow 24 的原生 DLL 与 torch 2.6 冲突**，且**依赖导入顺序**（torch 先加载才触发）。`import pyarrow` 单独不崩。
- **解决**：`pip install "pyarrow==18.1.0"`（datasets 4.0 要求 ≥15，18 稳定且不冲突）。

> **通用技巧**：段错误无 Python traceback 时，用 `python -u -c "import faulthandler; faulthandler.enable(); <imports>"` 拿到 C 层崩溃栈；再逐库二分定位。

---

## 二、CUDA 误报 OOM（实为多进程问题）

- **现象**：3B LoRA、batch=2，step 0 就 `RuntimeError: CUDA error: out of memory`，但 `nvidia-smi` 显示显存几乎全空。
- **定位**：完整 traceback 指向 `dataloader.py → multiprocessing spawn → torch reductions `_share_cuda_()``。
- **根因**：**Windows 用 spawn 起 DataLoader worker**，跨进程共享 CUDA 张量走 IPC handle 失败，被误报成 OOM。与真实显存无关。
- **解决**：配置里 **`dataloader_num_workers: 0`**（主进程加载数据）。这是 Windows 训练的必设项。

---

## 三、HuggingFace 下载问题

### 3.1 hf-mirror 镜像 + 新版 huggingface_hub 的 LFS 重定向失败
- **现象**：`hf_hub_download` 报 `Distant resource does not seem to be on huggingface.co` / `LocalEntryNotFoundError`。
- **根因**：huggingface_hub 1.23 改用 **Xet**，`HF_HUB_ENABLE_HF_TRANSFER` 已废弃；hf-mirror 对 LFS 文件的重定向指向官方 CDN，触发 hub 的 host 校验失败。
- **解决**：本机能直连官方 HF → **`.env` 注释掉 `HF_ENDPOINT`**，直连官方 + `HF_TOKEN`。（连不上官方的环境再启用镜像。）

### 3.2 `load_dataset(streaming=True)` 自动解析连不上
- **现象**：三个数据集全报连接错误，但 `list_repo_files` 能列出文件。
- **根因**：这些仓库是**单文件布局**（json/parquet/jsonl），streaming 自动解析脆弱。
- **解决**：改用 `hf_hub_download` 下指定文件 → 本地 `json/pandas` 解析，稳定可控。

### 3.3 gated 模型回退 ModelScope 拉了 6.4G 无用文件
- **现象**：Llama-3.2 在镜像下 HF 失败 → 回退 ModelScope → 下载 17 个文件含 6.43G `consolidated.00.pth`（Meta 原始格式，训练不用）。
- **解决**：杀进程、删残缺目录、修好 .env 后用 **HF + `allow_patterns=["*.safetensors","*.json","tokenizer*"]`** 只取需要的文件（12 个，6G）。

---

## 四、Windows / LLaMA-Factory CLI

### 4.1 `llamafactory-cli.exe train` 段错误
- **现象**：`.exe` 入口跑 train 崩溃（其实 1.1/1.2 的段错误也叠加在此）。
- **建议**：Windows 上统一用 **`python -m llamafactory.cli train ...`**，比 `.exe` wrapper 稳（路径含空格/撇号 `Ruibo's Desktop` 时尤甚）。

### 4.2 命令行覆盖 `eval_strategy=no` 被当成布尔 False
- **现象**：`ValueError: False is not a valid IntervalStrategy`。
- **根因**：LLaMA-Factory 的 `key=value` 覆盖用 **YAML 解析**，`no`/`yes` 被当布尔。
- **解决**：别在命令行传 `no`；把 eval 设置写进 yaml，或用 `eval_strategy=disable` 之外的合法值。冒烟测试直接沿用配置里的 `val_size`。

