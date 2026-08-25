# Week8 全链路 Pipeline 使用说明

> 交付物 · 任务书 41.3「主控脚本及使用说明」
> 适用版本：仓库 `main` 分支 · 实测环境 Windows 11 + RTX 4090 (24GB) + Python 3.12.10

---

## 〇、一分钟版本

```bash
# 仓库根目录
bash run_pipeline.sh --list           # 看看现在各段处于什么状态
bash run_pipeline.sh --quick          # 冒烟：小步数跑完全链路，验证装对了没有
bash run_pipeline.sh --skip-train     # 验收用：数据 + 评估两段，不占 GPU 几小时
bash run_pipeline.sh                  # 全链路：数据 → 训练 → 评估
```

根目录的 `run_pipeline.sh` 只是一层薄壳，真正的编排在 `Week8/scripts/run_pipeline.sh`。
薄壳做一件事：**把工作目录固定到仓库根**。项目里绝大多数脚本用
`Path(__file__).resolve().parents[2]` 自定位，但 LLaMA-Factory 的 `dataset_dir`
等少数配置是相对工作目录解析的，从别处 `cd` 进来直接跑会找不到数据集。

---

## 一、四个段分别做什么

| 段 | 脚本 | 输入 | 输出 | 典型耗时 |
|---|---|---|---|---|
| **data** | `step1_data_prep.py` | 三个原始数据源 | `Week8/data/*.json` + `dataset_info.json`<br>`Week8/deliverables/data_stats.{json,md}` | 30 s |
| **train** | `step2_train.sh` | 上一步的数据 + `sft_best.yaml` / `dpo_best.yaml` | `saves/week8/qwen/*` (LoRA)<br>`models/Qwen2.5-3B-week8-*-merged` | 数十分钟~数小时 |
| **eval** | `step3_eval.py` | 合并后的模型 | `Week8/deliverables/eval_summary.{csv,md}`<br>`eval_details/*.json`、`eval_answers/*.json` | 8 min（20 题） |
| **deploy** | `step4_deploy.sh` | 量化模型 | 常驻的 vLLM + Gradio 服务 | 启动 1~2 min，之后常驻 |

段与段之间只通过**文件**耦合，没有内存里的隐式状态。这意味着任何一段都可以单独跑、
单独调试、单独替换实现，只要它产出的文件契约不变。

---

## 二、参数速查

```
--skip-data          跳过数据准备
--skip-train         跳过训练（★ 验收推荐）
--skip-eval          跳过评测
--with-deploy        额外跑部署（默认关）
--only   <stage>     只跑一段：data / train / eval / deploy
--from   <stage>     从某段开始往后跑（用于失败后续跑）
--quick              冒烟模式：训练 2 步、评测 2 题
--train-stage <s>    透传给 step2：sft / dpo / all（默认 all）
--bench              评测时额外跑 CEval/CMMLU
--variant <v>        部署后端：fp16 / awq / gptq / vl（默认 awq）
--dry-run            只打印各段将执行的命令，不执行
--list               列出各段与产物状态后退出
```

### 三条最常用的组合

```bash
# 1) 我刚 clone 完，想确认环境装对了
bash run_pipeline.sh --quick

# 2) 我要复现交付文档里的评估数字，但不想重训模型
bash run_pipeline.sh --skip-train

# 3) 训练在 DPO 那一步挂了，修好之后从训练段续跑
bash run_pipeline.sh --from train --train-stage dpo
```

---

## 三、四条设计取舍（读完能省掉大部分踩坑）

### 3.1 为什么是「显式跳过」而不是「产物存在就自动跳过」

看起来更聪明的做法是自动判断产物是否存在。但四个段的代价差三个数量级
（30 s / 数小时 / 8 min / 常驻），自动跳过意味着**改了配置重跑时它会悄悄用旧产物**。
这类 bug 的症状是「我明明把 lr 从 1e-4 改成 2e-4 了，为什么分数一模一样」——
既难发现，发现之后又会让人怀疑之前所有的实验结论。

所以跳过哪一段永远是人写在命令行里的决定。脚本在开头把本次计划打出来让人核对：

```
[pipeline] 本次计划：
[pipeline]   step1 数据准备   执行
[pipeline]   step2 训练       跳过
[pipeline]   step3 评测       执行（bench=off）
[pipeline]   step4 部署       跳过（需 --with-deploy）
```

### 3.2 为什么部署段默认**不**跑

`step4` 起的是**常驻服务**，会一直占着显存和端口直到被显式停止。
一条「跑完就退出」的流水线突然变成「跑完还挂着两个后台进程」，对无人值守调用
（CI、夜间批处理）是灾难。所以 deploy 必须 `--with-deploy` 显式打开，
而且流水线结束时会明确打印怎么停：

```
[pipeline] ⚠ 服务仍在后台运行。停止： bash Week8/scripts/step4_deploy.sh --variant awq --stop
```

### 3.3 训练产物不存在时，评测段会退回评基座

`--skip-train` 时 `models/Qwen2.5-3B-week8-dpo-merged` 根本不存在。
如果评测段直接失败，验收标准 ❶ 要求的「数据 + 评估两段能跑通」就实现不了。
所以选择顺序是 **DPO 合并产物 → SFT 合并产物 → 基座**，并明确打印：

```
[pipeline] ⚠ 训练产物不存在，改评基座 .../Qwen2.5-3B-Instruct（这是第 3/4 周的对照基线，不是错误）
```

退回基座不是敷衍：基座就是第 3、4 周所有对比实验的对照基线，评它得到的分数
本身就有意义（见 §五的复现结果）。

### 3.4 失败即停，并且告诉你该去看哪里

数据没准备好还去训练、模型没合并出来还去评测——这类「带着错误往前冲」产生的
报错会指向完全无关的地方（报「模型目录不存在」，真正的原因是三步之前数据清洗挂了）。
所以一段失败就停，并打印排查顺序与续跑命令：

```
[pipeline] 流水线在 [eval] 中断。排查顺序：
[pipeline]   1) 上面这一段自己的报错（往上翻，别只看最后一行）
[pipeline]   2) 该段的日志：Week8/logs/
[pipeline]   3) 全流程日志：Week8/logs/pipeline_<run_id>.log
[pipeline]   4) README.md 的「常见问题」
[pipeline] 修好之后可以从这一段续跑： --from eval
```

---

## 四、各段的关键行为

### 4.1 data —— `step1_data_prep.py`

五步漏斗，每一步的进出数量都会打出来并写进 `data_stats.json`：

```
0_raw 4975 → 1_text_cleaned 4975 → 2_drop_empty 4975
      → 3_length_handled 4973 → 4_deduped 4684 → train 4216 / val 468
```

去重用 SimHash(64bit) + LSH(4 band × 16bit) + 汉明距离 ≤3，命中 289 条。

> ★ `sft_best.yaml` / `dpo_best.yaml` 里**没有** `val_size`。验证集在这一步就已经
> 显式切好，通过 `eval_dataset:` 传给 LLaMA-Factory。若同时设 `val_size`，
> LF 会在训练集上**再切一刀**形成二次划分，eval loss 就不是在同一份验证集上算的，
> 和第 3、4 周的数字失去可比性。

### 4.2 train —— `step2_train.sh`

依次执行 SFT → 合并 → DPO → 合并。OOM 时按五档阶梯自动降配重试：

| 档 | 动作 | 是否偏离原实验 |
|---|---|---|
| 1 | eval batch → 1 | 否（对训练数学零影响） |
| 2 | train batch 减半 **+ accum 加倍** | 否（等效 batch 不变） |
| 3 | 同上再来一次 | 否 |
| 4 | `cutoff_len` 减半 | ★ **是** |
| 5 | `cutoff_len` 再减半 | ★ **是** |

> ★ 第 2、3 档里 batch 减半和 accum 加倍**永远成对出现**，乘积恒等于配置文件里的
> 原值。只减 batch 不补 accum 的话，等效 batch 会从 16 掉到 8——每步看到的样本少
> 一半、梯度噪声变大，训出来的模型和「第 3 周的最优实验」根本不是一回事，
> 而流水线还会若无其事地把它当成最优模型交付。

第 4、5 档是真的改了实验（每步 token 数减半、长样本被多截），所以触发时日志打
★ 警告，`Week8/logs/retry_history.json` 里 `deviates_from_baseline=true`。
**宁可交付一个「知道自己不标准」的模型，也不要交付一个「以为自己标准」的。**

重试机制的注入式测试记录在 `Week8/logs/retry_history.json`（用假 python 制造
连续 OOM，验证五档阶梯逐级下降且历史文件写得对）。

### 4.3 eval —— `step3_eval.py`

两部分：

**① 20 题自定义集（必跑）** —— 题目在 `Week3/data/eval_questions.json`，
规则在 `Week8/configs/eval.yaml`，全在仓库里，不联网、不下数据包。
自动算 5 维分（准确性 0.30 / 完整性 0.25 / 逻辑性 0.20 / 安全性 0.15 / 格式 0.10，
权重与第 3 周人工评分卡完全一致）。

**② CEval/CMMLU 基准（可选，`--bench`）** —— 三级回退：

| 优先级 | 后端 | 条件 |
|---|---|---|
| ① | `opencompass` | `import opencompass` 成功 |
| ② | `llamafactory` | LF 自带 5-shot MCQA 评测器 |
| ③ | `unavailable` | 两条都不通，**如实记录原因** |

> ★ **绝不会出现第四种情况：跑不起来于是填一个看起来合理的数字。**
> 一份编造的 CEval 分数比一个空格危险得多——空格会让人去查，数字不会。
> CSV 里的 `⏳` 就是它该有的样子。
> 本机现状：OpenCompass 从未成功安装（第 3 周 Day15 起就卡着，
> `Week3/deliverables/OpenCompass评测分数表.md` 至今整张表都是 ⏳）。

### 4.4 deploy —— `step4_deploy.sh`

```bash
bash Week8/scripts/step4_deploy.sh                # awq 后端 + Gradio
bash Week8/scripts/step4_deploy.sh --variant gptq
bash Week8/scripts/step4_deploy.sh --status       # 只查健康
bash Week8/scripts/step4_deploy.sh --stop         # 停掉
```

> ★ **这个脚本要跨过 Windows / WSL 的边界**，因为两个服务不在同一个系统里：
> vLLM 没有 Windows 轮子，只能跑在 WSL2 的 `~/venvs/vllm`；Gradio 跑在 Windows 侧
> 的 `.venv`。脚本用 `uname -s` 判断自己在哪一侧——**注意 Git Bash(MSYS2) 也有
> `/proc/version`**（内容是 `MINGW64_NT-10.0-26200 ...`），只看这个文件会误判成
> Linux，然后去 nohup 一个 Windows 上根本不存在的 `vllm`。

> ★ **健康检查必须打 HTTP，不能看进程在不在。** vLLM 从进程起来到能接请求要
> 40~120 秒（加载权重 + 预分配 KV cache + 捕获 CUDA graph）。这段时间里进程活得
> 好好的，但任何请求都会被拒。所以脚本轮询 `GET /v1/models` 直到 200 或超时，
> 并每 15 秒打一次心跳，让人看见它在等而不是以为卡死了。

> ★ **curl 必须带 `--noproxy '*'`。** Windows 注册表里配了系统代理时，
> curl / httpx 都会把发往 `127.0.0.1` 的请求也塞进代理，代理转不了 localhost，
> 回 502。症状极具迷惑性：服务是好的，请求没到。

---

## 五、验收标准 ❶ 的实跑记录

> 「自动化 Pipeline 在干净环境中可无报错运行（至少包括数据准备和评估步骤）」

2026-08-25 实跑，日志原件存于 `Week8/deliverables/logs/`：

| 段 | 结果 | 耗时 | 日志 |
|---|---|---|---|
| data | ✓ | 32 s | `acceptance_run_data.log` |
| eval（基座，20 题） | ✓ | 505 s | `acceptance_run_eval.log` |

评测结果（自动 5 维）：

| 模型 | 准确性 | 完整性 | 逻辑性 | 安全性 | 格式 | 加权总分 |
|---|---|---|---|---|---|---|
| `base`（Qwen2.5-3B-Instruct） | 4.275 | 4.800 | 4.225 | 4.790 | 4.975 | **4.543** |

### ★ 一个意外的强证据：跨两周、跨脚本，20/20 题答案逐字节相同

这次生成的 20 条答案，与第 3 周 Day14 用 `Week3/code/eval_harness.py` 生成的
`answers_qwen_base.json` **逐题完全一致（20/20）**，因此五维分也分毫不差。

两次生成相隔约两周，用的是**两个独立编写的脚本**，中间还经历了一次换机重建环境。
能对上是因为三件事同时成立：贪心解码（`do_sample=False`）、同一份题目与顺序、
同一个 `max_new_tokens=512`，而依赖版本被 `requirements.txt` 钉死。

这件事的价值不在"分数好看"，而在于它把「可复现」从一句口号变成了一条可检验的断言：

```bash
.venv/Scripts/python.exe -c "
import json
a=json.load(open('Week3/deliverables/eval_answers/answers_qwen_base.json',encoding='utf-8'))['records']
b=json.load(open('Week8/deliverables/eval_answers/base.json',encoding='utf-8'))['records']
print(sum(1 for x,y in zip(a,b) if x['answer']==y['answer']), '/', len(a))
"
# → 20 / 20
```

---

## 六、常见问题

**Q1：`--quick` 和 `--dry-run` 有什么区别？**
`--dry-run` 一条命令都不执行，只打印将要执行什么，用来核对参数拼对了没有。
`--quick` 是**真的跑**，只是训练 2 步、评测 2 题——用来验证整条调用链通不通。
两者可以叠加，`--quick --dry-run` 打印的是冒烟模式下的命令。

**Q2：评测段报「空闲显存不足」怎么办？**
先 `nvidia-smi` 看是谁占着卡。常见的是上一次 `--with-deploy` 起的 vLLM 服务还挂着：

```bash
bash Week8/scripts/step4_deploy.sh --status     # 确认
bash Week8/scripts/step4_deploy.sh --stop       # 停掉
```

`--min-free-gb` 可以放宽阈值，但**放宽不会变魔术，只会让它在生成到一半时 OOM**，
把几分钟的生成全部浪费掉。

**Q3：为什么参数里有空格的路径会出问题？**
本仓库的真实路径是 `C:\Users\Ruibo's Desktop\SenceTime_Weeks1-5`，同时含**空格**
和**英文撇号**。撇号在 shell 里会开启单引号串。主控脚本里所有透传参数都攒在
**bash 数组**里而不是字符串里（`EVAL_ARGS=(--model "$EVAL_MODEL" ...)`），
展开时用 `"${EVAL_ARGS[@]}"` 不再二次分词。

> 这不是假想的问题：第一版就是用字符串攒的，实跑时 `Ruibo's` 和 `Desktop/...`
> 被拆成两个参数，argparse 报
> `unrecognized arguments: Desktop/SenceTime_Weeks1-5/models/...`——
> 报错指向 `step3_eval.py`，真正的错误却在主控脚本里，非常难找。

**Q4：Windows 上跑 `.sh` 用什么？**
Git Bash（随 Git for Windows 安装）。不要用 PowerShell 直接跑 `.sh`。
WSL 里也能跑，但那样 `PYTHON` 会指向 WSL 里的解释器，而训练栈装在 Windows 侧的
`.venv` 里——`pipeline.env` 会自动探测，实在不确定就看流水线开头打印的
`解释器: ...` 那一行。

**Q5：日志都在哪？**

```
Week8/logs/pipeline_<run_id>.log       整条流水线（tee，实时可看 + 事后可查）
Week8/logs/<stage>_<timestamp>.log     各训练段
Week8/logs/retry_history.json          OOM 降配重试历史
Week8/deliverables/logs/               验收实跑日志（进版本库）
```

---

## 七、相关文档

- `README.md` —— 环境安装、目录结构、十条常见问题
- `Week8/docs/Day40_数据与训练自动化.md` —— data / train 两段的实现细节
- `Week8/docs/Day41_评估与部署自动化.md` —— eval / deploy 两段的实现细节
- `Week8/docs/Day42_知识蒸馏.md` —— 蒸馏实验
- `Week8/reports/技术报告_Qwen2.5-3B全链路实践.md` —— 八章技术报告
