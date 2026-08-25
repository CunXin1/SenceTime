## 第 7 章 全链路自动化

前六章记录的是**六周的手工实验**：每一步都要人盯着，出了错要人判断，跑完要人整理。
本章讲的是把这条路径固化成一条可一键运行的 Pipeline 的过程，以及在固化过程中被迫
想清楚的几件事——因为**自动化最大的收益不是省人力，而是它逼你把每一个隐式决定
显式化**。手工跑的时候，"这次 batch 调小了一点""这次验证集是临时切的"都能靠记忆
糊过去；写进脚本时，这些糊过去的地方都会变成必须回答的问题。

### 7.1 架构：四段单向管道

#### 7.1.1 总体结构

```
                        ┌──────────────────────────────────────────┐
                        │        Week8/configs/pipeline.env        │
                        │  纯 POSIX sh：路径 / 数据集名 / 重试策略 │
                        └────────────────────┬─────────────────────┘
                                 每一段都 `.` 它，变量契约唯一
   ┌───────────────┬──────────────────┬──────┴───────────┬──────────────────┐
   ▼               ▼                  ▼                  ▼                  │
┌────────┐    ┌─────────┐        ┌─────────┐        ┌──────────┐            │
│ step1  │    │ step2   │        │ step3   │        │ step4    │            │
│ 数据   │───▶│ 训练    │───────▶│ 评测    │        │ 部署     │            │
│ .py    │    │ .sh     │        │ .py     │        │ .sh      │            │
└───┬────┘    └────┬────┘        └────┬────┘        └────┬─────┘            │
    │              │                  │                  │                  │
    ▼              ▼                  ▼                  ▼                  │
 Week8/data/   saves/week8/     eval_summary.csv    vLLM :8000               │
 dataset_      models/*-merged  eval_details/*      Gradio :7860             │
 info.json                      eval_answers/*      （常驻，需显式停）        │
    │              │                  │                                     │
    └──────────────┴──────────────────┴─────────────────────────────────────┘
                     段与段之间**只通过文件**耦合
                                 ▲
                        ┌────────┴─────────┐
                        │ run_pipeline.sh  │  --skip-* / --only / --from
                        │   主控编排        │  --quick / --dry-run / --list
                        └──────────────────┘
```

![图 7-1 Week8 全链路 Pipeline 架构](figs/fig7_1_pipeline_arch.png)

**图 7-1　Week8 全链路 Pipeline 架构**（上方 ASCII 图为终端可读版，同一结构）

图中两块底色对应两个操作系统：左侧浅蓝是 Windows 侧的 `.venv`（训练栈所在），
右侧浅橙是 WSL2 的 `~/venvs/vllm`。**这条边界是本项目架构上最特殊的一点**——
`step4_deploy.sh` 起 Gradio 用本地 `nohup`，起 vLLM 却要用 `wsl.exe -e bash -lc`
把命令投递到另一个操作系统里去，连停止服务时的 `kill` 也得同样投递
（WSL 命名空间里的 PID，Windows 的 taskkill 杀不掉）。

#### 7.1.2 为什么段与段之间只用文件耦合

四个段分别是 Python、Bash、Python、Bash，还要跨 Windows / WSL 两个操作系统
（原因见 §7.4）。任何基于内存对象、进程内状态或环境变量传递的耦合方式，
在这个异构组合里都不成立。

文件耦合的额外好处是**每一段都可以单独跑、单独调试、单独替换实现**，只要产出的
文件契约不变。第 7 章写完之后如果要把 step3 的打分器换成 LLM-as-judge，
只需要它仍然产出同样 schema 的 `eval_summary.csv`，主控和其余三段一行都不用改。

`pipeline.env` 是唯一的变量契约。它被刻意写成**纯 POSIX sh**——不用数组、
不用 `[[ ]]`、不用 `${VAR,,}`、不用 `local`——因为 step2/step3/step4 可能被
`bash` 也可能被 `sh`(dash) 加载，可能在 `set -u` 严格模式下。全文只用
`VAR="${VAR:-default}"` 这一种写法，它同时保证了「调用方可以用环境变量覆盖任何一项」
和「`-u` 下不会因未定义变量报错」。

### 7.2 使用示例

```bash
# 仓库根目录（run_pipeline.sh 是一层薄壳，负责把 CWD 固定到仓库根）
bash run_pipeline.sh --list           # 看看现在各段处于什么状态
bash run_pipeline.sh --quick          # 冒烟：小步数跑完全链路，验证装对了没有
bash run_pipeline.sh --skip-train     # 验收用：数据 + 评估两段
bash run_pipeline.sh                  # 全链路
bash run_pipeline.sh --from train --train-stage dpo   # 失败后续跑
```

`--list` 的输出直接回答「我现在在哪」：

```
段         脚本                        关键产物                              状态
---------------------------------------------------------------------------------
data       step1_data_prep.py          Week8/data/dataset_info.json          ✅
                                       Week8/deliverables/data_stats.json    ✅
train      step2_train.sh              models/...week8-sft-merged            —
                                       models/...week8-dpo-merged            —
eval       step3_eval.py               Week8/deliverables/eval_summary.csv   ✅
deploy     step4_deploy.sh             （常驻服务，无落盘产物）                —
```

### 7.3 四条设计取舍

#### 7.3.1 显式跳过，而不是「产物存在就自动跳过」

看起来更聪明的设计是自动判断产物是否存在。但四个段的代价差三个数量级
（32 秒 / 数小时 / 8 分钟 / 常驻），自动跳过意味着**改了配置重跑时它会悄悄用旧产物**。

这类 bug 的症状是「我明明把 lr 从 1e-4 改成 2e-4 了，为什么分数一模一样」——
既难发现，发现之后又会让人回头怀疑之前所有的实验结论。第 3 周做 14 组超参对比时，
如果有任何一组用了旧产物而没被发现，那一周的结论就全部作废了。

所以跳过哪一段永远是人写在命令行里的决定，脚本只负责忠实执行，并在开头把本次计划
打出来让人核对。

#### 7.3.2 部署段默认不跑

`step4` 起的是**常驻服务**，会一直占着显存和端口直到被显式停止。一条「跑完就退出」
的流水线突然变成「跑完还挂着两个后台进程」，对无人值守调用是灾难——下一个任务
拿到的是一张已经被吃掉 20GB 的卡，而它并不知道为什么。

所以 deploy 必须 `--with-deploy` 显式打开，且流水线结束时明确打印停止命令。

#### 7.3.3 训练产物不存在时退回评基座

`--skip-train` 时训练产物根本不存在。如果评测段直接失败，验收标准 ❶ 要求的
「数据 + 评估两段能跑通」就实现不了。所以选择顺序是
**DPO 合并产物 → SFT 合并产物 → 基座**，并明确打印这不是错误。

退回基座不是敷衍：基座就是第 3、4 周所有对比实验的对照基线，评它得到的分数本身就
有意义——§7.6 里那条最强的证据正是这样得到的。

#### 7.3.4 失败即停，并告诉你该去看哪里

数据没准备好还去训练、模型没合并出来还去评测——这类「带着错误往前冲」产生的报错会
指向完全无关的地方（报「模型目录不存在」，真正原因是三步之前数据清洗挂了）。
所以一段失败就停，并打印排查顺序与续跑命令。

### 7.4 跨 Windows / WSL 边界

这是本项目区别于大多数教程型 Pipeline 的地方：**两个服务不在同一个操作系统里**。

- **vLLM 没有 Windows 轮子**（官方只发 Linux wheel），只能跑在 WSL2 的 `~/venvs/vllm`
- **Gradio 跑在 Windows 侧的 `.venv`**（第 5 周的多模态资源、第 3 周的题集都在
  Windows 文件系统上，来回跨 `/mnt/c` 反而慢）

于是 `step4_deploy.sh` 必须先回答「我在哪一侧」，再决定用哪种启动方式：
本地 `nohup`，还是 `wsl.exe -e bash -lc` 投递。

#### ★ 一个实测踩到的判断错误

第一版的判断逻辑是：

```bash
if grep -qi microsoft /proc/version; then echo wsl
elif [ -r /proc/version ]; then echo linux      # ← 错在这里
else echo windows; fi
```

在 Git Bash 里它判成了 `linux`，然后去 `nohup` 一个 Windows 上根本不存在的 `vllm`。
原因是 **MSYS2 也提供 `/proc/version`**：

```
$ cat /proc/version
MINGW64_NT-10.0-26200 version 3.6.9-b4195d69.x86_64 (@runnervmlu3mh) ...
```

正确写法是先用 `uname -s` 把 `MINGW*` / `MSYS*` / `CYGWIN*` 挑出去，再用
`/proc/version` 里的 `microsoft` 认 WSL。

这个错误有代表性：**「文件存在」和「文件存在且含义是我以为的那个」是两件事。**
跨平台脚本里的特征检测尤其容易在这里翻车，因为兼容层的存在感恰恰体现在
「它把你以为只有 Linux 才有的东西也提供了」。

#### ★ 路径里的撇号与空格

仓库真实路径是 `C:\Users\Ruibo's Desktop\SenceTime_Weeks1-5`，**同时含空格和英文撇号**。
撇号在 shell 里会开启单引号串，不加双引号的话任何一处展开都会把后面的整行吞掉。

投递进 WSL 的命令串里，路径被包在双引号内，bash 把整串作为**一个 argv** 传给
`wsl.exe`，WSL 侧的 `bash -lc` 解析时撇号在双引号内部是字面量。已实测：
`cd` 进带撇号的目录并成功 `import vllm`（0.27.1）。

### 7.5 三个在实跑中才暴露的 Bug

自动化脚本有个特点：**很多错误只在真跑的时候才出现**，dry-run、语法检查、
代码审查都发现不了。本节记录三个，因为它们各自代表一类。

#### 7.5.1 参数攒进字符串而不是数组

```bash
EVAL_ARGS="--model $EVAL_MODEL --tag $EVAL_TAG"
run_stage eval "$PYTHON" "$SCRIPTS/step3_eval.py" $EVAL_ARGS
```

实跑立刻炸：

```
step3_eval.py: error: unrecognized arguments: Desktop/SenceTime_Weeks1-5/models/Qwen2.5-3B-Instruct
```

`$EVAL_ARGS` 展开时按空格分词，`Ruibo's` 和 `Desktop/...` 被拆成两个参数。

**这个 bug 的教学价值在于报错的位置**：错误信息出自 `step3_eval.py` 的 argparse，
真正的错误却在主控脚本里。跨脚本的错误定位是最费时间的一类——因为第一反应总是
去看报错的那个文件。

正确写法是 bash 数组，每个元素是一个 argv，`"${arr[@]}"` 展开后不再二次分词。

#### 7.5.2 `curl` 的退出码与输出同时发生

```bash
"$CURL" -s -o /dev/null -w '%{http_code}' --max-time 5 "$1" 2>/dev/null || echo "000"
```

连不上时 curl **既**把 `000` 打到 stdout（`-w` 照样输出），**又**返回非零退出码。
`|| echo "000"` 让两者都发生，函数返回 `"000000"`，后面所有
`[ "$code" = "200" ]` 的比较全部失效。

**失效得很安静**：状态永远显示 ❌，看起来完全像「服务没起来」。
如果当时手边正好没起服务，这个 bug 可能要等到很久以后才被发现。

修法是把 `|| echo "000"` 换成 `; true`。

#### 7.5.3 健康检查看进程而不是看 HTTP

vLLM 从进程起来到能接请求要 **40~120 秒**（加载权重 + 预分配 KV cache +
捕获 CUDA graph）。这段时间里进程活得好好的，但任何请求都会被拒。

如果 step4 起完就报「部署成功」，紧接着的冒烟请求必然失败，而失败原因看起来像
「服务挂了」。所以脚本轮询 `GET /v1/models` 直到 200 或超时，并**每 15 秒打一次心跳**
——让人看见它在等，而不是以为脚本卡死了。

这三个 bug 有个共同点：**它们都不会让脚本崩溃，只会让它给出错误的结论。**
崩溃是好事，错误结论才是真正的成本。

### 7.6 验收 ❶ 的实跑记录

> 「自动化 Pipeline 在干净环境中（提供依赖清单）可无报错运行
> （至少包括数据准备和评估步骤）」

2026-08-25 实跑 `bash run_pipeline.sh --skip-train`，日志原件存于
`Week8/deliverables/logs/`：

**表 7-1　验收实跑结果**

| 段 | 结果 | 耗时 | 日志 |
|---|---|---|---|
| data | ✓ | 32 s | `acceptance_run_data.log` |
| train | 按参数跳过 | — | — |
| eval（基座，20 题，逐题贪心生成） | ✓ | 505 s | `acceptance_run_eval.log` |
| deploy | 默认跳过 | — | — |

评测结果：

**表 7-2　基座模型在 20 题集上的自动 5 维分**

| 模型 | 准确性 | 完整性 | 逻辑性 | 安全性 | 格式 | 加权总分 |
|---|---|---|---|---|---|---|
| `base`（Qwen2.5-3B-Instruct） | 4.275 | 4.800 | 4.225 | 4.790 | 4.975 | **4.543** |

#### ★ 一个意外的强证据：跨两周、跨脚本，20/20 题答案逐字节相同

这次生成的 20 条答案，与第 3 周 Day14 用 `Week3/code/eval_harness.py` 生成的
`answers_qwen_base.json` **逐题完全一致（20/20）**，因此五维分也分毫不差。

两次生成相隔约两周，用的是**两个独立编写的脚本**，中间还经历了一次换机重建环境
（第 6 周）。能对上是因为三件事同时成立：贪心解码（`do_sample=False`）、
同一份题目与顺序、同一个 `max_new_tokens=512`，而依赖版本被 `requirements.txt`
钉死到 `+cu124` 级别。

这件事有两层价值：

1. **它把「可复现」从一句口号变成了一条可检验的断言**，任何人都能用一行命令验证：

   ```bash
   .venv/Scripts/python.exe -c "
   import json
   a=json.load(open('Week3/deliverables/eval_answers/answers_qwen_base.json',encoding='utf-8'))['records']
   b=json.load(open('Week8/deliverables/eval_answers/base.json',encoding='utf-8'))['records']
   print(sum(1 for x,y in zip(a,b) if x['answer']==y['answer']), '/', len(a))"
   # → 20 / 20
   ```

2. **它顺带证明了 `step3_eval.py` 与 `eval_harness.py` 在生成语义上等价**
   （同一个 chat template、同一套生成参数），因此 Week8 的分数可以和第 3 周的横向
   对比——这本来是需要单独论证的一件事，现在被这个巧合一次性解决了。

### 7.7 自动化逼出来的三个「本该早就想清楚」的问题

本章开头说「自动化最大的收益是逼你把隐式决定显式化」。具体是这三个：

#### 7.7.1 验证集到底是谁切的

`sft_best.yaml` 里**没有** `val_size`——验证集在 step1 就显式切好，通过
`eval_dataset:` 传入。如果同时设 `val_size`，LLaMA-Factory 会在**已经切过的训练集上
再切一刀**，形成二次划分：实际训练样本变成 4216 × 0.9 = 3794 条，eval loss 是在
多切出来的 422 条上算的。

**它不报错、不警告、训练照跑、曲线照出，只是数字悄悄换了含义。**
手工跑的时候这件事被"反正每次都这么跑"糊过去了；写进流水线时必须在两者中选一个，
才发现原来一直有两个验证集。

#### 7.7.2 OOM 降配时等效 batch 变了没有

等效 batch = `per_device_train_batch_size × gradient_accumulation_steps`。
第 3 周的冠军超参是在等效 batch = 16 下调出来的。

手工遇到 OOM 时的自然反应是「把 batch 调小一点再跑」——如果不同时把 accumulation
补上，等效 batch 就从 16 掉到 8，训出来的模型和「第 3 周的最优实验」根本不是一回事，
**而流水线还会若无其事地把它当成最优模型交付**。

所以 40.3 的降配阶梯里，batch 减半和 accum 加倍**永远成对出现**，乘积恒等于配置
文件里的原值。而降 `cutoff_len` 因为真的改了实验（每步 token 数减半、长样本被多截），
被放在阶梯末端，触发时打 ★ 警告并在 `retry_history.json` 里标
`deviates_from_baseline=true`。

> **宁可交付一个「知道自己不标准」的模型，也不要交付一个「以为自己标准」的。**

#### 7.7.3 跑不起来的评测该记什么

任务书要求「运行 OpenCompass 并解析结果」。本机的真实情况是：OpenCompass 从来没有
成功跑起来过——第 3 周 Day15 就卡住了，那张分数表至今整张都是 ⏳。

手工阶段这件事可以一直挂着。写进流水线时必须回答：跑不起来的时候，那一格填什么？

答案是**三级回退 + 如实记录**：`opencompass` → `llamafactory` → `unavailable`，
每一级都把「用了哪个后端、为什么」写进 CSV 的两列。**绝不允许出现第四种情况：
跑不起来于是填一个看起来合理的数字。**

一份编造的 CEval 分数比一个空格危险得多——空格会让人去查，数字不会。
而且它会一路污染下游：技术报告引用它、结论建立在它上面、别人复现不出来时会先怀疑
自己的环境。同理，缺失值写 `⏳` 而不是 `0` 或空字符串：`0` 会被下游当成
「考了 0 分」参与平均。

### 7.8 本章小结

1. **架构上**：四段单向管道，只通过文件耦合，`pipeline.env` 是唯一变量契约。
   这个结构是被异构环境（Python/Bash、Windows/WSL）逼出来的，但它换来了
   「任何一段都可单独替换」的好处。
2. **取舍上**：显式跳过而非自动跳过、部署段默认关、训练产物缺失时退回基座、
   失败即停并给出续跑命令——四条都指向同一个原则：
   **让流水线的行为可预测，而不是让它显得聪明。**
3. **实跑上**：三个 bug 都不会让脚本崩溃，只会让它给出错误结论。
   崩溃是好事，错误结论才是真正的成本。
4. **方法论上**：自动化的收益不止是省人力。把六周的手工流程写成脚本，
   逼出了三个一直被糊过去的问题——验证集是谁切的、降配时等效 batch 变了没有、
   跑不起来的评测该记什么。**这三个问题的答案，比 Pipeline 本身更有价值。**

### 本章引用

[1] Zheng Y, Zhang R, Zhang J, et al. LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models. ACL 2024 System Demonstrations, 2024.
[2] Kwon W, Li Z, Zhuang S, et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP, 2023.
[3] Charikar M S. Similarity Estimation Techniques from Rounding Algorithms. STOC, 2002.
