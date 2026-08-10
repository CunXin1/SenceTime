# LoRA 微调报告(Day 5)

> 在 Mac(M4 Pro / MPS)上用 LLaMA-Factory 完成两次 LoRA 微调,验证训练管线并演示效果。
> 脚本:`code/finetune.py`(训练)、`code/compare_finetune.py`(前后对比)。

---

## 一、两次微调概况

| 项目 | ① alpaca_zh_demo | ② identity_penguin |
|---|---|---|
| 语料 | 中文通用指令(100/1000 条) | 自我认知(91 条) |
| 目标 | 通用指令遵循 | 改身份为 Penguin AI |
| 训练耗时 | **3 分 48 秒** | **1 分 39 秒** |
| 步数 | 75(3 轮) | ~75(3 轮) |
| 最终 loss | 1.445(avg) | 收敛 |
| **adapter 大小** | **35 MB** | **35 MB** |
| 基座大小(未变) | 2.9 GB | 2.9 GB |

**关键点**:LoRA 只存 35MB 的 adapter(基座 2.9GB 不动),仅占 **1.2%** —— 这就是 LoRA 高效的本质。

---

## 二、微调前后对比(identity_penguin)

同样问"你是谁",基座 vs 微调后:

### 微调前(基座 Qwen2.5-1.5B)
```
Q: 你是谁?
A: 我是 Qwen,由阿里云开发的超大规模语言模型。

Q: Who are you?
A: I am Qwen, an AI language model developed by Alibaba Cloud.

Q: 你叫什么名字?是谁开发的?
A: 我是由阿里云研发的语言模型,我叫通义千问。
```

### 微调后(+ identity_penguin LoRA)
```
Q: 你是谁?
A: 您好,我是 Penguin AI,一个由 Ruibo Sun 开发的人工智能助手。

Q: Who are you?
A: I am Penguin AI, an AI assistant developed by Ruibo Sun.

Q: 你叫什么名字?是谁开发的?
A: 您好,我是 Penguin AI,由 Ruibo Sun 开发。
```

**结论**:仅用 91 条身份语料 + 100 秒训练,模型的自我认知从"Qwen / 阿里云"成功改写为 **"Penguin AI / Ruibo Sun"**,中英文都生效。

---

## 三、验证要点

1. **LLaMA-Factory 能在 Mac MPS 上训练** —— 普通 LoRA(非 QLoRA)跑通,无报错(验收 ❹ ✅)。
2. **LoRA 只存增量 adapter**(35MB),基座不变;推理时"基座 + adapter"叠加。
3. **语料决定微调方向**:通用指令语料 → 提升指令遵循;身份语料 → 改自我认知。
4. **训练可视化**:`training_loss.png`(loss 曲线)+ `runs/`(TensorBoard 日志)。

---

## 四、如何切换到 7B / QLoRA(4090)

`code/finetune.py` 顶部改两个变量即可:
```python
MODEL = "./models/Qwen2.5-7B-Instruct"   # 换 7B
QLORA = 4                                  # 4-bit QLoRA(仅 CUDA / 4090)
```
Mac 无 CUDA 只能普通 LoRA;7B QLoRA 按任务要求在 4090 上跑。

## 五、部署压缩(AWQ)

`code/export_awq.py`:先合并 LoRA 到基座(Mac 可做),再 AWQ 4-bit 量化(需 CUDA / 4090)。

## 附:交付物
- `alpaca_loss.png` / `identity_loss.png` —— 两次训练的 loss 曲线
- adapter 权重在 `saves/*/`(35MB 各)
