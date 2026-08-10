# Week3 FAQ / 踩坑记录

> 沿用 Week2 惯例：把本周遇到的问题、原因与解法沉淀下来，供以后（和别人）复用。

## Q1：训练无故比上周慢 65%（17.2s/步 vs 10.4s/步），配置完全一样？

**原因**：Chrome / Edge WebView / 微信 / Steam 等桌面程序挂在 4090 上，Windows WDDM
调度把 GPU 时间片分给了显示任务。识别特征：`nvidia-smi` 显示利用率 98% 但功耗只有
333W/450W（"忙而不饱"）。

**解法**：训练前关掉占 GPU 的桌面程序。清场后自动恢复 10.1s/步。
**验证方法**：loss 曲线与上周逐点一致（seed 固定），证明慢的不是训练本身。

## Q2：想再快点——加大 batch size 或关梯度检查点有用吗？

**实测没用，反而更慢**（各跑 2000 样本基准）：

| 方案 | s/步 | 峰值显存 |
|---|---|---|
| bs1 + 检查点（基线）| **10.12** | 13.3 GB |
| bs2 + 检查点 | 10.38 | 18.1 GB |
| bs4 + 检查点 | 12.00 | 23.9 GB |
| bs1 关检查点 | 12.00 | 23.8 GB |

原因：①开了 packing 后每步已是满 2048 token 的饱和计算，加大 micro-batch 只增加
显存不增加吞吐；②关检查点让激活显存逼近 24GB 上限，分配开销反噬了重计算的节省。
**教训：提速前先做基准测试，直觉经常是错的。**

## Q3：`pip install liger-kernel` 在 Windows 上报 ResolutionImpossible？

**原因**：liger-kernel 声明依赖 `triton`（Linux 包名），Windows 的 PyPI 上没有这个包
（Windows 版叫 `triton-windows`，但提供同名 `triton` 模块）。

**解法**：
```powershell
pip install triton-windows          # 先装，提供 triton 模块
pip install liger-kernel --no-deps  # 跳过依赖解析
```

## Q4：Windows 上装 FlashAttention-2 要自己编译吗？

不用。社区维护预编译轮子（[mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)），
按 `cuda版本 + torch版本 + python版本 + win_amd64` 挑完全匹配的文件名即可：

```powershell
pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.19/flash_attn-2.8.3%2Bcu124torch2.6-cp312-cp312-win_amd64.whl"
```

注意版本三要素必须与 `python -c "import torch; print(torch.__version__)"` 完全一致。

## Q5：两组实验能不能在一张 4090 上并行跑？

不能。单组 bf16 LoRA 峰值 13.3GB，两组 26.6GB > 24GB。QLoRA（4-bit）能塞下，
但量化基座改变了实验对象，对比实验的结论就失效了——**科学性优先于速度**。

## Q6：中途换加速内核（SDPA → FA2/Liger），前后的实验还可比吗？

**loss 可比，耗时/显存不可比。** FA2 和 Liger 都是精确计算（不是近似 attention），
loss 差异在 bf16 浮点重排噪声级；但速度和显存是内核属性，结果汇总表中需要分口径
标注哪些组用了哪种内核。

## Q7：eval_steps 设 200 为什么从来没触发过中途评估？

总步数只有 ~108（packing 后），到不了 200。按总步数的 1/3~1/2 设置（本周用 50），
保证每次训练至少 2~3 个 eval 点——组C（轮数）实验判断过拟合全靠 eval loss 走势。

## Q8：OpenCompass 显示跑完了（exit 0），但汇总表全是 "-"？

**原因**：项目路径 `C:\Users\Ruibo's Desktop\...` 含**撇号和空格**。OpenCompass 用
Linux 风格的单引号拼子进程命令，Windows cmd 不认单引号，命令在撇号处碎裂
（debug 日志里反复出现 `'C:\Users\Ruibo's' is not recognized...`），推理静默失败，
但主流程照常写出空汇总——**exit 0 不等于出分，必须检查 summary 里有没有数字**。

**解法**：在无特殊字符的路径建评测工作区，模型/数据用目录联接挂载（无需管理员）：
```powershell
New-Item -ItemType Directory C:\oc
New-Item -ItemType Junction C:\oc\models -Target "<repo>\models"
New-Item -ItemType Junction C:\oc\data   -Target "<repo>\data"
py -3.12 -m venv C:\oc\venv   # venv 不可搬迁，必须原地重建
# 装 opencompass + CUDA torch 后，一律 cd C:\oc 再跑评测
```
**教训**：Windows 上做深度学习，项目路径只用 `字母数字-_`；另外 pip 在 Windows
默认装 CPU 版 torch，装完必须验 `torch.cuda.is_available()`（本周也踩了）。

## Q9：OpenCompass 官方只支持 Linux，Windows 怎么办？

本周策略（已验证到安装环节）：
1. 独立 venv（`.venv-oc`）避免污染训练环境 → `opencompass 0.5.3` 在 py3.12 + Win11
   安装成功；
2. 数据不从 HF 拉，用官方整包 OpenCompassData-core 解压到 `./data/`；
3. 跑不通就退 `llamafactory-cli eval`（原生 CEval/CMMLU，可直接挂 adapter），
   分数表注明评测框架即可。排障表见 `Week3/code/run_opencompass.md` §4。
