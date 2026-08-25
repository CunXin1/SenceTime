# Day40 — 数据与训练自动化

> 任务书 40.1 / 40.2 / 40.3
> 交付：`step1_data_prep.py`、`step2_train.sh`、数据统计报告

---

## 一、40.1 `step1_data_prep.py` —— 数据准备

### 1.1 五步漏斗

每一步的进出数量都会打印并写进 `Week8/deliverables/data_stats.json`。
2026-08-25 实跑结果：

```
=== SFT 清洗漏斗 ===
  0_raw                4975
  1_text_cleaned       4975      （改动 593 条：全角空格、零宽字符、重复标点）
  2_drop_empty         4975
  3_length_handled     4973      （截断 2，超长丢弃 2）
  4_deduped            4684      （SimHash 去重命中 289）
  5_train              4216
  5_val                 468
=== DPO 漏斗 ===
  0_raw                1221
  1_cleaned            1221
  2_deduped            1219      （按 prompt 去重命中 2）
  3_train              1097
  3_val                 122
总耗时 30.18 s（去重 8.18 s 占大头）
```

来源构成：`alpaca_gpt4_zh` 2000 / `coig_pc` 1976 / `sharegpt_zh` 999。

### 1.2 为什么去重用 SimHash + LSH，而不是精确哈希或两两比对

- **精确哈希**（MD5/SHA）只能抓完全相同的样本。实际数据里的重复大多是
  「同一道题换了个提问方式」「多了一个句号」——精确哈希一条都抓不到。
- **两两比对**是 O(n²)：4975 条要比 1237 万次，每次还要算编辑距离，跑不完。
- **SimHash(64bit) + LSH 分桶**是 O(n)：把 64 位签名切成 4 个 16 位 band，
  只有至少一个 band 完全相同的样本对才进入汉明距离比较。实测 8.18 s 跑完，
  命中 289 条近似重复。

汉明距离阈值取 **≤3**：64 位签名上距离 3 意味着约 95% 的特征位一致。
阈值放到 5 会开始误伤（不同题目但句式高度模板化的样本被判重复），
收到 1 又会漏掉大部分只差一两个词的重复。

### 1.3 ★ 划分为什么必须在这一步做完，而不是交给 LLaMA-Factory

任务书写的是「随机划分，比例 9:1」。LLaMA-Factory 有个 `val_size` 参数看起来
正好能干这件事——但两者**不能同时用**：

`sft_best.yaml` / `dpo_best.yaml` 里**没有** `val_size`，验证集在 step1 就切好了，
通过 `eval_dataset:` 显式传入。如果同时设 `val_size`，LF 会在**已经切过的训练集上
再切一刀**，形成二次划分：

- 实际训练用的样本变成 4216 × 0.9 = 3794 条
- eval loss 是在这多切出来的 422 条上算的，而不是我们准备的那 468 条
- 于是 Week8 的 eval loss 和第 3、4 周的数字**失去可比性**，而且看不出来

这个坑的隐蔽之处在于：它不报错、不警告，训练照跑，曲线照出，只是数字悄悄换了含义。

### 1.4 两种格式同时产出

Alpaca 格式（`train.json` / `val.json`）与 ShareGPT 格式
（`train_sharegpt.json` / `val_sharegpt.json`）同时产出，并在 `dataset_info.json`
里各自注册。原因：多轮对话数据在 Alpaca 格式下会被压平成单轮，
丢掉轮次结构；而 Alpaca 格式在 LF 里的很多路径上更成熟。同时留两份，
后续换 stage 或换模板时不用重跑数据准备。

---

## 二、40.2 `step2_train.sh` —— 训练自动化

### 2.1 链路

```
SFT 训练 → 合并 → DPO 训练 → 合并
```

```bash
bash Week8/scripts/step2_train.sh                 # 全链路
bash Week8/scripts/step2_train.sh --stage sft     # 只跑 SFT(+合并)
bash Week8/scripts/step2_train.sh --dry-run       # 只打印命令
bash Week8/scripts/step2_train.sh --quick         # 冒烟（max_steps=2）
```

日志自动落 `Week8/logs/<stage>_<timestamp>.log` 并同时 tee 到终端。

### 2.2 ★ 覆盖参数必须写成 `key=value`，不能写成 `--key value`

LLaMA-Factory 的 `hparams/parser.py:90-93`：当 `argv[1]` 是 `.yaml` 时，
剩余参数走 `OmegaConf.from_cli(sys.argv[2:])` —— 那是 OmegaConf 的
**dotlist 语法**，只认 `key=value`。写成 `--per_device_train_batch_size 2`
会被 OmegaConf 当成一个叫 `"--per_device_train_batch_size"` 的键而报错。

这跟直接用 `llamafactory-cli` 传参的写法不一样，很容易踩——两种写法在文档里
都出现过，但适用的入口不同。

### 2.3 ★ 为什么不用 `set -e`

这个脚本的核心逻辑就是「命令失败之后继续做事」（判断是不是 OOM、降配、重跑）。
`set -e` 会在第一次训练失败时直接把脚本杀掉，整套重试机制永远不会被执行到。

用的是 `set -uo pipefail` + 逐处显式检查返回码。**`pipefail` 是必须的**：
训练命令要 `| tee` 到日志，没有 pipefail 的话 `$?` 拿到的是 tee 的返回码
（几乎永远是 0），训练崩了也会被当成成功。

### 2.4 Windows 相关的两条硬约束

- **只能用 `"$PYTHON" -m llamafactory.cli`**，不能用 `.venv/Scripts/llamafactory-cli.exe`
  —— 第 2 周 Day10 实测该入口在 Windows 上段错误（console_script 的 wrapper
  与 torch DLL 加载顺序冲突）。
- **`dataloader_num_workers=0`** —— Windows 的 spawn 启动方式会让 worker 进程各自
  初始化 CUDA 上下文，触发 CUDA IPC 的假 OOM。

---

## 三、40.3 错误重试机制

### 3.1 五档降配阶梯

按「对原实验的扰动从小到大」排序，逐档往下走。不适用的档（比如 batch 已经是 1
还要求减半）会被自动跳过，不浪费一次重试名额。

| 档 | 动作 | 是否偏离原实验 |
|---|---|---|
| 1 | `per_device_eval_batch_size` → 1 | 否——对训练数学**零影响**，只省评估时的峰值显存 |
| 2 | train batch 减半 **+ accum 加倍** | 否——等效 batch 不变 |
| 3 | 同上再来一次 | 否 |
| 4 | `cutoff_len` 减半 | ★ **是** |
| 5 | `cutoff_len` 再减半 | ★ **是** |

### 3.2 ★ 取舍一：重试时**等效 batch 必须保持不变**

等效 batch = `per_device_train_batch_size × gradient_accumulation_steps`。
第 3 周的冠军超参是在「等效 batch = 16」下调出来的，第 4 周是 8。

如果重试时只把 per_device batch 减半、不补 accumulation，等效 batch 就从 16 掉到 8：
每步看到的样本少一半 → 梯度噪声变大 → 有效学习率相当于被改了 → 训出来的模型和
「第 3 周的最优实验」根本不是一回事，**而流水线还会若无其事地把它当成最优模型交付**。

所以降配阶梯里，batch 减半和 accum 加倍**永远成对出现**，乘积恒等于配置文件里的
原值，并在日志里打出来核对。

### 3.3 ★ 取舍二：降 `cutoff_len` 是最后一档，且会被显式标成「已偏离原实验」

降 cutoff 是真的改了实验：它既减少每步的 token 数（等效 batch 在「序列」口径上不变、
在「token」口径上减半），又会把长样本截得更短、改变数据分布。

但它确实是 batch 已经降到 1 之后唯一还剩的手段。所以放在阶梯末端，触发时日志里打
★ 警告，`retry_history.json` 里 `deviates_from_baseline=true`。

> **宁可交付一个「知道自己不标准」的模型，也不要交付一个「以为自己标准」的。**

### 3.4 ★ OOM 判定：退出码非 0 **且** 日志里出现显存不足的特征串

只看退出码不行——语法错、数据集没注册、路径写错也都是非 0，那些重试一万次也不会好，
降配重试只会把真正的错误信息埋在三层重试日志底下，让排查从「读一条报错」变成
「在四份日志里找哪一份是第一现场」。

### 3.5 注入式测试：怎么在不制造真 OOM 的前提下验证重试逻辑

真的把卡跑到 OOM 来测试重试，既慢（每次要等训练加载完模型）又不可控
（OOM 发生在哪一步取决于当时别的进程占了多少）。

做法是把 `$PYTHON` 换成一个假解释器，它把收到的 argv 打进日志，然后打印一条
真实的 `torch.OutOfMemoryError` 回溯并返回 1。这样可以**确定性地**验证：

- 五档阶梯是不是按顺序触发
- 每档的覆盖参数拼得对不对
- `deviates_from_baseline` 标记有没有在第 4 档正确翻转
- `retry_history.json` 写出来是不是合法 JSON 数组

实跑记录（`Week8/logs/retry_history.json`）：

| attempt | 档位 | overrides | deviates |
|---|---|---|---|
| 0 | baseline（配置文件原值） | （无） | false |
| 1 | eval batch 2→1 | `per_device_eval_batch_size=1` | false |
| 2 | ★ cutoff_len→1024 | `per_device_train_batch_size=1 gradient_accumulation_steps=16 per_device_eval_batch_size=1 cutoff_len=1024` | **true** |
| 3 | ★ cutoff_len→512 | 同上，`cutoff_len=512` | **true** |

> 注意 attempt 2 的 overrides：`batch=1` 与 `accum=16` 同时出现，
> 乘积 16 与配置文件里的等效 batch 相同——这正是 §3.2 要保证的性质，
> 在注入式测试里被直接验证了。

### 3.6 ★ 历史文件的写入不能依赖 `$PYTHON`

`retry_history.json` 是用 sed 从 JSONL 明细渲染出来的，而不是调 Python。
理由很实际：注入式测试会把 `$PYTHON` 换成假命令，如果历史文件的写入依赖那个
被替换掉的解释器，测试就测不到真实的写入路径了。

两个文件的分工：
- `retry_history.jsonl` —— 追加型明细，**永不重写**
- `retry_history.json` —— 每次从 JSONL 重新渲染成合法 JSON 数组，供人和 step3 读

---

## 四、模型合并

`merge_model.py --config Week8/configs/merge_sft.yaml`

合并前检查磁盘余量（`MIN_FREE_GB=10`，一个 3B bf16 模型约 6GB）。
不够就早失败——总比写到一半 ENOSPC、留下一个半截模型目录强，
那种目录还会被后续步骤当成「模型已存在」而跳过。
