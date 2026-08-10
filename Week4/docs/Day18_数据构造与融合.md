# Day18：人工构造与开源数据融合（执行记录）

> 交付物：合并后的偏好数据集 `Week4/data/dpo/dpo_pairs.json`（1221 条）+《偏好数据集统计.md》。

## 今天做了什么

1. **自建 221 条**（`make_self_built_pairs.py` → `self_built_pairs.json`）：
   安全性 80（60 正向拒答覆盖 10 类风险 + 20 反过度拒绝）/ 事实正确性 40 / 完整性 36 /
   有用性 35 / 格式 30。内容全部手写，脚本只负责组装/编号/校验，可一键复现。
2. **开源数据抽取**（`build_preference_data.py`）：
   - UltraFeedback（`llamafactory/ultrafeedback_binarized`）抽 500 条（英）；
   - DPO-En-Zh-20k（`llamafactory/DPO-En-Zh-20k` 的 `dpo_zh.json`）抽 500 条单轮中文对；
   - 用 `random.Random(42)` 抽样，可复现。
3. **统一为 LLaMA-Factory sharegpt 偏好格式**并注册 `week4_dpo_pairs`（`ranking: true`），
   剥离元数据成 `dpo_pairs.json`（训练用）+ `pairs_meta.json`（统计用），出 2 张统计图。

## 产出文件

| 文件 | 说明 |
|---|---|
| `Week4/data/self_built_pairs.json` | 自建 221 条（带 pref_type/sub_type 元数据） |
| `Week4/data/dpo/dpo_pairs.json` | 合并 1221 条，仅 conversations/chosen/rejected |
| `Week4/data/dpo/pairs_meta.json` | 同序来源/类型元数据 |
| `Week4/data/dpo/dataset_info.json` | LLaMA-Factory 注册表（ranking: true） |
| `Week4/deliverables/偏好数据集统计.md` + `pref_dist.png` + `pref_length.png` | 统计 |

## 关键设计决策

1. **中英语言平衡**：SFT 语料与评测题全为中文，直接混 500 条英文对会拉偏中文表现，故加 500 条
   中文对（DPO-En-Zh-20k）平衡。UltraFeedback 提供通用有用性/正确性偏好基础。
2. **只取单轮中文对**：DPO-En-Zh-20k 含约 18% 多轮样本，为与自建对同构、避免多轮上下文干扰，
   只取 `conversations` 长度为 1 的样本。
3. **校验规则自动执行**：非空 / chosen≠rejected / 长度合理性（安全类豁免）/ 自建 prompt 去重。
   开源数据里出现过 chosen==rejected 的脏样本，在加载阶段过滤掉。

## 插曲

- 开源 DPO-En-Zh-20k 里有个别 chosen 与 rejected 完全相同的样本，第一次校验直接 assert 失败。
  解法：对开源数据在 load 阶段做 `chosen!=rejected` 过滤（自建数据仍保留硬断言，因为是我们自己写的）。

## 明日衔接

Day19 写 DPO 配置模板并启动 3 组对比训练。
