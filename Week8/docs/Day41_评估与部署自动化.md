# Day41 — 评估与部署自动化

> 任务书 41.1 / 41.2 / 41.3
> 交付：`step3_eval.py`、`step4_deploy.sh`、`run_pipeline.sh` 及使用说明
> 使用说明单独成篇：`Week8/docs/Pipeline使用说明.md`

---

## 一、41.1 `step3_eval.py` —— 自动评估

### 1.1 两段结构

| 段 | 是否必跑 | 依赖 |
|---|---|---|
| 20 题自定义集 + 自动 5 维打分 | **必跑** | 全在仓库里，不联网 |
| CEval / CMMLU 基准 | 可选（`--bench`） | 需要评测框架 + 约 1.6GB 数据包 |

**为什么主干是自定义题集、基准评测反而是增强项**：验收标准 ❶ 要求
「干净环境可无报错运行（至少包括数据准备和评估步骤）」。评估这一环的主干必须是
那个一定跑得通的部分。20 题集完全自给自足——题目在 `Week3/data/eval_questions.json`，
打分规则在 `Week8/configs/eval.yaml`，两者都在版本库里。

---

### 1.2 ★ 最重要的一条：跑不起来时不会编个分数出来

任务书写的是「运行 OpenCompass（CEval/CMMLU）并解析结果」。本机的真实情况是：

> **OpenCompass 从来没有在这台机器上成功跑起来过。**
> 第 3 周 Day15 就卡住了，`Week3/deliverables/OpenCompass评测分数表.md` 至今
> 整张表都是 ⏳，排障过程记在 `Week3/code/run_opencompass.md`。

所以基准评测部分实现成**三级回退**，每一级都把「用了哪个后端、为什么」写进 CSV 的
`bench_backend` / `bench_note` 两列：

| 优先级 | 后端 | 触发条件 |
|---|---|---|
| ① | `opencompass` | `import opencompass` 成功 |
| ② | `llamafactory` | LF 自带的 5-shot MCQA 评测器可用 |
| ③ | `unavailable` | 两条路都不通，**如实记原因** |

> ★ **绝不允许出现第四种情况：「跑不起来，于是填一个看起来合理的数字」。**
> 一份编造的 CEval 分数比一个空格危险得多——空格会让人去查，数字不会。
> 而且它会一路污染下游：技术报告引用它、结论建立在它上面、别人复现不出来时
> 会先怀疑自己的环境。CSV 里的 `⏳` 就是它该有的样子。

同理，`bench_to_columns()` 里缺失值一律写 `⏳` 而**不是 0 或空字符串**：
0 会被下游当成「考了 0 分」参与平均；空字符串在 Excel 里和「真的是 0」长得太像。

### 1.3 ★ 解析对象是 `results.log` 文件，不是 stdout

LF 的 `Evaluator._save_results()`（`eval/evaluator.py:139-154`）把
`f"{category_name:>15}: {100*np.mean(...):.2f}"` **同时** print 到 stdout
**和**写进 `<save_dir>/results.log`。

以文件为准：stdout 里混着 tqdm 的进度条回车，在 Windows 控制台上经常把分数行冲掉半截。

另一个必须注意的点：`save_dir` **必须不存在**。LF 的
`EvaluationArguments.__post_init__`（`hparams/evaluation_args.py:58-60`）
对已存在的 `save_dir` 直接 raise。所以脚本里带时间戳生成目录名。

---

### 1.4 自动 5 维打分器 `auto_score.py`

#### 它是什么，更重要的是它不是什么

人工打分考察的是「这个回答对不对、好不好」，那需要真正读懂内容。本脚本做不到，
它做的是**表面特征匹配**：答案里有没有出现正确的最终值、有没有推理链的形状、
有没有烂尾/复读。

因此它是人工评分的 **proxy（代理指标）**，不是替代品。正当用途只有两个：

1. **回归护栏** —— 同一套题、同一套规则，模型改版后分数掉了能立刻发现；
2. **流水线自动化** —— 41.3 要求无人值守跑完 data→train→eval→deploy，
   这一环不能等人来读 100 份答卷。

#### 为什么是规则法而不是 LLM-as-judge

LLM 裁判分更准，但它 ① 要占一张卡（本周多个任务并行，必须礼让 GPU）
② 不确定（同一份答卷两次跑分不同，回归护栏就废了）③ 无法解释（掉分说不出原因）。

规则法反过来：零显存、完全可复现、每一分都能追到具体哪条规则。
这三点正好是「流水线里的自动评估」最需要的。代价是天花板低——它读不懂内容。

#### 对齐质量（拿第 3 周 100 条真实人工打分做的回归）

```
=== 模型级：自动 vs 人工（加权总分）===
模型                          自动    人工     差    自动排名/人工排名
qwen_base                     4.54   4.54   0.01   1 / 1
qwen_r8_lr1e-4_ep5            4.20   4.34  -0.14   2 / 2
qwen_r8_lr2e-4_ep3            4.10   4.19  -0.09   4 / 3
qwen_r32_lr1e-4_ep3           4.13   4.05   0.08   3 / 4
qwen_r8_lr1e-4_ep3            3.96   4.01  -0.05   5 / 5

模型级 Spearman rho = 0.90（n=5，仅第 3/4 名互换）
题级 rho = 0.699，题级平均绝对偏差 MAE = 0.376
```

维度级明细：

| 维度 | 自动 | 人工 | 差 | 题级 rho |
|---|---|---|---|---|
| 准确性 | 3.82 | 3.99 | −0.17 | 0.877 |
| 完整性 | 4.23 | 4.49 | −0.26 | 0.259 |
| 逻辑性 | 4.07 | 3.97 | +0.10 | 0.405 |
| 安全性 | 4.65 | 4.38 | +0.27 | 0.846 |
| 格式 | 4.74 | 4.56 | +0.18 | 0.537 |

**这个精度足以当回归护栏，不足以下「A 比 B 好 0.1 分」这种结论。**
准确性与安全性两维的题级 rho 分别是 0.877 / 0.846，可信度最高；
完整性只有 0.259——它本质上是在数「要点覆盖率」，而人在判完整性时会考虑
「对这道题来说什么算完整」，规则法接近不了。CSV 因此同时给 5 个维度分而不只是总分：
**维度分掉在哪一维，比总分掉了多少更有诊断价值。**

#### ★ 一处建模方式的修正（逻辑性维度）

第一版的逻辑性是 `1 + 4 × 加权信号量`（数连接词、数步骤号、数等式）。
实测全模型均值 2.54，人工是 3.97，**系统性低了 1.43 分**，是五维里最大的偏差，
调系数救不回来。

根因是建模方式反了：人工评分卡的锚点是「5 = 完整自洽 / 3 = 有明显跳步或一处矛盾 /
2 = 多处矛盾」——**人是默认给高分再扣的**；而线性加分模型默认给低分再加，
于是所有语气平实、不爱用连接词的回答（尤其是代码题）都被冤枉。

改成同样的「默认自洽、见到破绽才扣」结构后，均值回到 3.9 附近。

> 这条经验的普遍性：**自动指标要对齐的不只是人的分数，还有人的打分结构。**

#### ★ 两档匹配与 `eval.yaml` 的必要性

`eval_questions.json` 的 `reference` 字段是**写给人看的散文**，例如 math-03 的
「`1/(1/6+1/4) = 1/(5/12) = 2.4 小时（2 小时 24 分钟）`」，里面混着推导过程的
中间量（1、6、4、5、12）和真正的最终答案（2.4）。

如果无差别地要求模型答案命中 reference 里出现的**所有**数字，一个正确但简洁的回答
（「2.4 小时」）会被判低分，而一个啰嗦但算错的回答反而占便宜——「准确性」这一维
彻底失真。

所以 `eval.yaml` 把每题的 reference **人工蒸馏**成结构化要点：

- `final` —— 最终答案，决定**准确性**
- `support` —— 过程要点，决定**完整性**
- `anti` —— 已知的典型错答，命中直接扣分

蒸馏只依据 `reference` 本身，**没有看过任何模型答卷**——否则就是在拟合测试集。

代价必须说明：这张表是**题集专属**的，换一套题就得重写。所以实现了两档——
有 `answer_key` 的题走精确档，没有的题自动退回**通用档**（从 reference 里机械抽取
数字与关键词，精度低但零配置），CSV 里标出每题用了哪一档。

#### ★ `scope: conclusion` —— 一类必须单独处理的误判

有些题的正确答案**就写在题干里**：

- reason-02「照片里的人是我父亲的儿子」——「我父亲的儿子」是题干原话，全文匹配必然误判为答对
- reason-04「100 天后是星期几」——推理过程里会写「1 天后是星期四，2 天后是星期五」，全文匹配会被举例骗过
- reason-06「5 分钟」——题干本身就写着 5 分钟

所以这些题的 `final` 标了 `scope: conclusion`，只在**结论区**匹配。
不这么做的话，这几题会给所有模型无差别送分，把维度分整体抬高而失去区分度。

---

## 二、41.2 `step4_deploy.sh` —— 自动部署

### 2.1 ★ 这个脚本要跨过 Windows / WSL 的边界

两个服务**不在同一个操作系统里**：

- **vLLM 没有 Windows 轮子**（官方只发 Linux wheel），只能跑在 WSL2 的 `~/venvs/vllm`
- **Gradio 跑在 Windows 侧的 `.venv`**（第 5 周的多模态资源、第 3 周的题集都在
  Windows 文件系统上，来回跨 `/mnt/c` 反而慢）

于是脚本做的事是：**在哪边就用哪边的启动方式**——

- 脚本本身跑在 WSL/Linux → 直接 `nohup` 起 vllm
- 脚本本身跑在 Git Bash → 用 `wsl.exe -e bash -lc` 把命令投递进 WSL

### 2.2 ★ `detect_side()` 的判断顺序不能反

第一版是这么写的：

```bash
if [ -r /proc/version ] && grep -qi microsoft /proc/version; then echo wsl
elif [ -r /proc/version ]; then echo linux          # ← 错在这里
else echo windows; fi
```

实测（2026-08-25）在 Git Bash 里判成了 `linux`，然后去 `nohup` 一个 Windows 上
根本不存在的 `vllm`。原因是 **MSYS2 也提供 `/proc/version`**：

```
$ cat /proc/version
MINGW64_NT-10.0-26200 version 3.6.9-b4195d69.x86_64 (@runnervmlu3mh) ...
```

正确写法是先用 `uname -s` 把 `MINGW*` / `MSYS*` / `CYGWIN*` 挑出去，
再拿 `/proc/version` 里的 `microsoft` 认 WSL。

### 2.3 ★ 健康检查必须打 HTTP，不能看进程在不在

vLLM 从进程起来到能接请求要 **40~120 秒**（加载权重 + 预分配 KV cache +
捕获 CUDA graph）。这段时间里进程活得好好的，但任何请求都会连接被拒。

如果 step4 起完就报「部署成功」，紧接着的冒烟请求必然失败，
而失败原因看起来像「服务挂了」，实际只是没等它起完。

所以脚本轮询 `GET /v1/models` 直到 200 或超时（`--timeout`，默认 300 s），
并且**每 15 秒打一次心跳**——让人看见它在等，而不是以为脚本卡死了。

### 2.4 ★ `curl` 必须带 `--noproxy '*'`

第 7 周踩过整整一轮：Windows 注册表里配了系统代理，httpx / curl 都会读环境里的
`http_proxy` 并把发往 `127.0.0.1` 的请求也塞进代理，代理转不了 localhost，回 502。

症状极具迷惑性——**服务是好的，请求没到**。

Gradio 那边靠 `app.py` 里的 `NO_PROXY` + `httpx.Client(trust_env=False)` 解决；
这里健康检查用的是 curl，对应的开关是 `--noproxy '*'`。

### 2.5 ★ `http_code()` 末尾必须是 `; true`，不能是 `|| echo "000"`

```bash
"$CURL" -s -o /dev/null -w '%{http_code}' --noproxy '*' --max-time 5 "$1" 2>/dev/null; true
```

连不上时 curl **既**把 `000` 打到 stdout（`-w '%{http_code}'` 照样输出），
**又**返回非零退出码。写成 `|| echo "000"` 会让两者都发生，函数返回 `"000000"`，
后面所有 `[ "$code" = "200" ]` 的比较全部失效——而且失效得很安静：
状态永远显示 ❌，看起来像「服务没起来」。

实测记录：`--status` 打出 `HTTP=000000`。

### 2.6 ★ PID 文件与跨系统的 kill

PID 文件写在 `Week8/logs/` 而不是 `/tmp`——`/tmp` 在 Windows 和 WSL 里是两个目录，
脚本又要跨边界，用 `/tmp` 必然对不上。

更关键的是：**WSL 侧 vLLM 的 PID 是 WSL 命名空间里的 PID**，Windows 的 taskkill
杀不掉它。所以 `--stop` 时同样要把 `kill` 投递进 WSL 去执行。

### 2.7 ★ 路径里的撇号：为什么 `wsl.exe -e bash -lc "$inner"` 是对的

仓库路径是 `C:\Users\Ruibo's Desktop\SenceTime_Weeks1-5`，同时含空格和英文撇号。
投递进 WSL 的命令串里，路径被包在**双引号**内：

```
cd "/mnt/c/Users/Ruibo's Desktop/SenceTime_Weeks1-5" && ...
```

bash 把整串作为**一个 argv** 传给 `wsl.exe`，WSL 侧的 `bash -lc` 解析时撇号在双引号
内部是字面量。已实测：`cd` 进带撇号的目录并成功 `import vllm`（0.27.1）。

`--dry-run` 打印时故意**不**给这串套单引号——套上会变成一条自己都跑不了的命令，
照着复制粘贴的人必然踩坑。

---

## 三、41.3 `run_pipeline.sh` —— 主控

详见 `Week8/docs/Pipeline使用说明.md`。这里只记一条实跑中发现的 bug。

### ★ 透传参数必须攒进 bash 数组，不能攒进字符串

第一版写的是：

```bash
EVAL_ARGS="--model $EVAL_MODEL --tag $EVAL_TAG"
run_stage eval "$PYTHON" "$SCRIPTS/step3_eval.py" $EVAL_ARGS
```

实跑立刻炸：

```
step3_eval.py: error: unrecognized arguments: Desktop/SenceTime_Weeks1-5/models/Qwen2.5-3B-Instruct
```

`$EVAL_ARGS` 展开时按空格分词，而仓库路径是 `C:/Users/Ruibo's Desktop/...`——
`Ruibo's` 和 `Desktop/...` 被拆成了两个参数。

**报错信息指向 `step3_eval.py`，真正的错误却在主控脚本里**，这类跨脚本的错误定位
是最费时间的。正确写法：

```bash
EVAL_ARGS=(--model "$EVAL_MODEL" --tag "$EVAL_TAG")
[ "$QUICK" = "1" ] && EVAL_ARGS+=(--quick)
run_stage eval "$PYTHON" "$SCRIPTS/step3_eval.py" "${EVAL_ARGS[@]}"
```

数组的每个元素是一个 argv，加引号展开 `"${arr[@]}"` 后不再二次分词。
`TRAIN_ARGS` / `DEPLOY_ARGS` 同步改成数组——它们当前不含路径，但依赖
「这个变量恰好不含空格」是一种迟早会失效的假设。

---

## 四、实跑证据

2026-08-25，`bash run_pipeline.sh --skip-train`，日志原件存于
`Week8/deliverables/logs/`：

| 段 | 结果 | 耗时 |
|---|---|---|
| data | ✓ | 32 s |
| eval（基座，20 题，逐题贪心生成） | ✓ | 505 s |

### 一个意外的强证据

这次生成的 20 条答案，与第 3 周 Day14 用 `Week3/code/eval_harness.py` 生成的
`answers_qwen_base.json` **逐题完全一致（20/20）**，五维分也分毫不差
（4.275 / 4.800 / 4.225 / 4.790 / 4.975，总分 4.543）。

两次生成相隔约两周，用的是**两个独立编写的脚本**，中间还经历了一次换机重建环境。
能对上是因为三件事同时成立：贪心解码、同一份题目与顺序、同一个
`max_new_tokens=512`，而依赖版本被 `requirements.txt` 钉死到 `+cu124` 级别。

这件事顺带证明了 `step3_eval.py` 的生成路径与 `eval_harness.py` 在语义上等价
（同一个 chat template、同一套生成参数），因此 Week8 的分数可以和第 3 周的
横向对比——这本来是需要单独论证的一件事。
