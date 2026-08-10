# Day15：OpenCompass 评测操作手册（含 Windows 排障与兜底方案）

> 目标：用 CEval + CMMLU 客观评测 **Qwen 基座 + Qwen 最优 SFT 模型**，产出分数表。
> Goal: score the Qwen base + the best Qwen SFT model on CEval + CMMLU.
>（Day12 范围缩减后只训 Qwen；Llama 相关命令保留在注释里备用。）

## 0. 环境隔离（为什么单独建 .venv-oc）

OpenCompass 对 `transformers/datasets` 等有自己的版本约束，直接装进 `.venv`
可能破坏 LLaMA-Factory 的训练环境（Week3 实验还要用）。因此：

```powershell
# 独立环境，与训练环境完全隔离 / isolated from the training venv
py -3.12 -m venv .venv-oc
.\.venv-oc\Scripts\python.exe -m pip install --upgrade pip
.\.venv-oc\Scripts\python.exe -m pip install opencompass
```

⚠️ 官方推荐 Python 3.10 + Linux；本机只有 3.12 + Windows，属于**非官方支持组合**，
遇到依赖编译失败优先看 §4 排障，仍不通则走 §5 兜底。

## 1. 数据集准备（CEval / CMMLU）

OpenCompass 不从 HF 拉数据，用自己的整包：

```powershell
# 在仓库根目录执行；数据会解压到 ./data（OpenCompass 默认相对工作目录找 data/）
Invoke-WebRequest https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip -OutFile OpenCompassData-core.zip
Expand-Archive OpenCompassData-core.zip -DestinationPath .
```

也可手动下载后解压，保证目录形如 `data/ceval/...`、`data/cmmlu/...`。

## 2. 冒烟测试（先单科目验证能出分）

```powershell
# 只跑 CEval 的一个子集 + 基座模型，几分钟内验证全链路
.\.venv-oc\Scripts\opencompass.exe `
  --datasets ceval_gen `
  --hf-type chat `
  --hf-path models/Qwen2.5-3B-Instruct `
  --max-num-workers 1 `
  --debug
```

`--debug`：单进程直跑、日志直出，Windows 上排障必开。

## 3. 正式评测（4 个模型 × 2 数据集）

最优 SFT 模型须先用 `Week3/configs/merge_best_qwen.yaml` 合并导出
（OpenCompass 直接吃 HF 目录，不认 LoRA adapter）：

```powershell
# ⚠️ Windows 上 export 用 python -m（llamafactory-cli.exe 会段错误，Week2 Day10 FAQ）
.\.venv\Scripts\python.exe -m llamafactory.cli export Week3/configs/merge_best_qwen.yaml
# （若恢复 Llama 实验再执行）
# .\.venv\Scripts\python.exe -m llamafactory.cli export Week3/configs/merge_best_llama.yaml
```

然后逐个评测（顺序跑，避免显存互挤）：

```powershell
$models = @(
  "models/Qwen2.5-3B-Instruct",
  "models/Qwen2.5-3B-week3-best-merged"
  # , "models/Llama-3.2-3B-Instruct", "models/Llama-3.2-3B-week3-best-merged"
)
foreach ($m in $models) {
  .\.venv-oc\Scripts\opencompass.exe `
    --datasets ceval_gen cmmlu_gen `
    --hf-type chat --hf-path $m `
    --max-out-len 512 --batch-size 8 --max-num-workers 1
}
```

结果在 `outputs/default/<时间戳>/summary/` 下的 csv/txt，
把各模型分数手工汇总进 `Week3/deliverables/OpenCompass评测分数表.md`。

## 4. Windows 排障清单

| 症状 | 处置 |
|---|---|
| ★ exit 0 但汇总全是 "-"（无分数） | 路径含撇号/空格导致子进程命令碎裂（见 FAQ Q8）。在 `C:\oc` 建干净工作区 + junction 挂 models/data + 原地重建 venv，`cd C:\oc` 再跑 |
| torch 是 CPU 版（评测极慢/不用 GPU） | Windows pip 默认装 CPU torch；重装 `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`，验证 `torch.cuda.is_available()` |
| `Qwen2Tokenizer has no attribute batch_encode_plus` | transformers 5.x 移除了老 API，OpenCompass 0.5.3 不兼容；回退 `pip install transformers==4.56.2` |
| 中文/编码报错 UnicodeDecodeError | 先 `$env:PYTHONUTF8 = "1"` 再跑 |
| 依赖编译失败（如 faiss、某些 C 扩展） | 这些是可选依赖；`pip install opencompass` 核心包不需要，报错时看缺的是哪个包，能跳过就跳过 |
| 卡在多进程/partitioner | 加 `--max-num-workers 1 --debug` 强制单进程 |
| 找不到数据集 | 确认工作目录下有 `data/ceval`、`data/cmmlu`（§1 的解压位置） |
| 显存不足 | `--batch-size` 降到 4 或 2 |

## 5. 兜底方案：LLaMA-Factory 自带 eval（保底能出分）

LLaMA-Factory 原生支持 CEval/CMMLU（MCQA 5-shot，与 OpenCompass 口径略有差异，
分数表中需注明评测框架）。用训练同款 `.venv`，Windows 上稳定：

```powershell
# 基座 / base（--task: ceval_validation / cmmlu_test / mmlu_test）
.\.venv\Scripts\llamafactory-cli.exe eval `
  --model_name_or_path models/Qwen2.5-3B-Instruct `
  --template qwen --task ceval_validation --lang zh `
  --n_shot 5 --batch_size 4 --trust_remote_code true `
  --save_dir saves/week3/eval/qwen_base_ceval

# 最优 SFT：无需合并，直接挂 adapter / adapters work directly, no merge needed
.\.venv\Scripts\llamafactory-cli.exe eval `
  --model_name_or_path models/Qwen2.5-3B-Instruct `
  --adapter_name_or_path saves/week3/qwen/<best_run> `
  --template qwen --task ceval_validation --lang zh `
  --n_shot 5 --batch_size 4 --trust_remote_code true `
  --save_dir saves/week3/eval/qwen_best_ceval
```

Llama 模型把 `--template` 换成 `llama3`。每次运行在 `save_dir` 生成
`results.json`（按学科的准确率 + 平均分）。

## 6. 交付物格式（OpenCompass评测分数表.md）

| 模型 | CEval (avg) | CMMLU (avg) | 评测框架 | 备注 |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct（基座） | | | OpenCompass / LF eval | |
| Qwen2.5-3B **最优 SFT** | | | | 与基座的差值标出 |

> 预期现象：中文指令 SFT 对 CEval/CMMLU 这类知识型选择题**提升有限甚至持平**
> （SFT 学的是对话风格而非新知识）；若明显下降需在周报里分析灾难性遗忘。
