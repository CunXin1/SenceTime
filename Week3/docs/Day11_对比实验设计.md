# Day11：对比实验设计（完成记录）

> 本文是 Day11 的工作说明；正式交付物见 `Week3/deliverables/实验计划表.md`。
> 实验矩阵为 Qwen 单模型 4 组（1 共享基线 + 每组 1 变体），见《实验计划表.md》§2。

## 今天做了什么

1. **设计 3 组核心对比实验**（任务 11.1）：LoRA 秩（8/32/64）、学习率（5e-5/1e-4/2e-4）、
   轮数（2/3/5），全部以 Week2 的 `r8 / lr1e-4 / ep3` 为基线。
2. **固定随机种子与数据集**（任务 11.2）：所有配置显式 `seed: 42`，数据固定为 Week2 清洗后的
   `week2_sft_alpaca`（4684 条），验证集切分因同种子而完全一致。
3. 写了模板与生成器，**14 个实验配置全部由脚本生成**，杜绝手改 yaml 引入的变量污染。

## 产出文件

| 文件 | 说明 |
|---|---|
| `configs/template_qwen.yaml` / `template_llama.yaml` | 实验模板（fork 自 Week2 配置，含 `${...}` 占位符，中英双语注释） |
| `code/gen_configs.py` | 生成器：7 个唯一实验 × 2 模型 → 14 个 yaml + 清单 |
| `configs/exp/*.yaml`（14 个） | 可运行配置，头部注明实验名/组别/变量取值 |
| `configs/exp/experiments.json` | 机器可读实验清单，Day12 的 `run_experiments.py` 直接消费 |
| `deliverables/实验计划表.md` | **正式交付**：控制变量表 + 实验矩阵 + 每组假设 + 判定规则 |

## 关键设计决策（为什么这么做）

1. **9 组名义实验去重为每模型 7 次训练**：基线 `r8_lr1e-4_ep3` 在三组中完全相同，
   只训一次、三组共用，省 4 次训练约 80 分钟。对比表仍按 9 组呈现。
2. **α 随秩同步（α=2r）**：LoRA 旁路输出乘 α/r 回加主干。若固定 α=16，则 r=64 时
   等效缩放只有 0.25，秩实验会被缩放系数干扰；保持 α/r≡2 才是干净的单变量实验。
3. **`eval_steps` 从 200 改为 50**：Week2 总步数仅 ~108，eval_steps=200 导致训练中途
   从未评估过。50 保证每次训练有 2~3 个 eval 点——组C（轮数）实验判断过拟合全靠
   eval loss 曲线。此改动对训练本身无影响（评估不参与梯度）。
4. **双模型 = 结论的重复验证**：Qwen 与 Llama 同矩阵各跑一遍；若两边最优超参一致，
   结论具备跨架构泛化性（呼应 Week2 的跨架构对比传统）。
5. **模板 + 生成器而非手写 14 份 yaml**：变量以外的字段物理上不可能不一致；
   复现只需 `python Week3/code/gen_configs.py`。

## 复现方式

```bash
# 在仓库根目录 SenceTime_Week1/ 下
.venv/Scripts/python.exe Week3/code/gen_configs.py
# → Week3/configs/exp/ 生成 14 个 yaml + experiments.json
```

## 明日衔接（Day12-13）

`run_experiments.py` 读取 `experiments.json` 顺序跑 14 组：每组记录墙钟耗时、
nvidia-smi 峰值显存 → `run_meta.json`；支持断点续跑。预算 ≈ 4.5 小时 GPU。
