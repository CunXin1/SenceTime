# Week8 Day40 数据统计报告

> 本文件由 `Week8/scripts/step1_data_prep.py` 自动生成，勿手改。

- 生成时间：2026-08-25T15:47:52　脚本版本：`step1_data_prep.py v1.0.0`
- 原始数据：`Week2/data/unified/alpaca_all.json`
- 分词后端：`HF:Qwen2.5-3B-Instruct`　cutoff：2048 token
- 划分：train:val = 90%:10%，seed=42
- 总耗时：15.44 秒

## 一、SFT 数据清洗漏斗

| 阶段 | 样本数 | 说明 |
|---|---|---|
| 0 原始 | 4975 | `alpaca_all.json` 直接读入 |
| 1 文本清洗 | 4975 | HTML 标签 / 控制字符 / 空白规范；改动 593 条，不丢样本 |
| 2 空值过滤 | 4975 | instruction 或 output 为空则丢 |
| 3 长度处理 | 4973 | 截断 output 2 条；instruction 本身超长丢弃 2 条 |
| 4 模糊去重 | 4684 | SimHash(64bit)+LSH，命中重复 **289** 条 |
| 5 划分 | train 4216 / val 468 | 随机 9:1，seed=42 |

## 二、长度分布

| 集合 | 口径 | n | min | median | p90 | max | mean |
|---|---|---|---|---|---|---|---|
| train | token | 4216 | 7 | 170.0 | 523.5 | 2040 | 226.0 |
| train | char | 4216 | 7 | 286.0 | 949.5 | 4198 | 402.3 |
| val | token | 468 | 8 | 170.0 | 445.2 | 1110 | 210.5 |
| val | char | 468 | 14 | 286.5 | 856.2 | 1870 | 379.0 |

## 三、来源分布

| 来源 | 原始 | train | val |
|---|---|---|---|
| alpaca_gpt4_zh | 2000 | 1786 | 213 |
| coig_pc | 1976 | 1545 | 150 |
| sharegpt_zh | 999 | 885 | 105 |

## 四、DPO 偏好数据

| 阶段 | 样本数 |
|---|---|
| 0 原始 | 1221 |
| 1 清洗（去空 / 去 chosen==rejected） | 1221 |
| 2 按 prompt 去重 | 1219 |
| 3 划分 | train 1097 / val 122 |

- 去重命中：**2** 条

| 集合 | 口径 | n | min | median | p90 | max | mean |
|---|---|---|---|---|---|---|---|
| dpo_train | token | 1097 | 11 | 321 | 716.4 | 2101 | 362.7 |
| dpo_val | token | 122 | 15 | 239.5 | 684.4 | 1141 | 322.3 |

## 五、耗时（秒）

| 阶段 | 秒 |
|---|---|
| load | 0.02 |
| clean_text | 0.04 |
| length | 1.37 |
| dedup | 8.22 |
| split | 0.0 |
| dpo | 1.63 |

## 六、产物

| 文件 | 格式 | 用途 |
|---|---|---|
| `train.json` / `val.json` | Alpaca | SFT 训练 / 验证（sft_best.yaml 直接吃） |
| `train_sharegpt.json` / `val_sharegpt.json` | ShareGPT | 同一批样本的多轮编码，备用 |
| `dpo_train.json` / `dpo_val.json` | ShareGPT + ranking | DPO 训练 / 验证 |
| `dataset_info.json` | — | LLaMA-Factory 注册表，6 个数据集 |

> ★ 注意：`sft_best.yaml` / `dpo_best.yaml` 里**没有** `val_size`。
> 验证集由本脚本显式切好，通过 `eval_dataset:` 传给 LLaMA-Factory。
> 两者不能并存：同时写 LF 会抛 `Cannot specify val_size if eval_dataset is not None`；
> 只留 `val_size` 则本脚本切的 val 集完全不被使用，LF 会在 train 上另切一刀（静默的二次划分）。
