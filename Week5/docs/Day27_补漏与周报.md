# Day27：补漏与周报

## 一、补漏清单（对照任务书逐条核）

| 任务书条目 | 要求 | 实际 | 状态 |
|---|---|---|---|
| 22.1 选型 | 8GB→2B / 16GB+→7B | RTX 4090 24GB → 7B/8B 两个模型 | ✅ 偏差已在 README §一 说明理由 |
| 22.2 下载 + qwen-vl-utils | 模型下载确认 | `Day22_模型下载确认.md`（含参数拆解/显存/延迟实测） | ✅ |
| 22.3 5 张图片 | 表格/风景/Logo/手写/UI | **6 张**（多一张业务图表），带 `ground_truth.json` | ✅ 超额 |
| 23.1 5 类 × 5 图 | 25 条记录 | **60 条**（5 类 × 6 图 × 2 模型） | ✅ 超额 |
| 23.2 好/坏场景记录 | — | `Day23_能力边界分析.md` §四 | ✅ |
| 24.1 Hook 抽第 20 层 Cross-Attention | — | **两个模型都没有 cross-attn**，改抽 self-attn 的 text→image 子块 | ✅ 方法论纠正已记录 |
| 24.2 热力图 ≥3 张 | 3 | **6 张**（3 组 × 单层/逐层） | ✅ 超额 |
| 25.1 10 组图片-问题-假答案 | 10 | **53 条探针 × 2 模型 = 106 条**，分 3 类 | ✅ 超额 |
| 25.2 幻觉率统计 | 产生不存在物体的比例 | 物体幻觉率 + 抗误导率 + 立场翻转率 + Yes-ratio | ✅ 超额 |
| 26.1 200 条图文指令数据 | 200 | 200 训练 + 20 独立留出 | ✅ |
| 26.2 LoRA 微调（冻结 ViT） | — | `freeze_vision_tower` + `freeze_multi_modal_projector`，可训练 0.4845% | ✅ |
| 27.2 周报 | — | `第5周_多模态实践报告.md` / `.docx` | ✅ |

## 二、Day27 实际做的三件事

1. **重跑 Day26 训练**。第一次在 step 25/69 中断（原因与两处配置错误见 `Day26_VLM轻量微调.md` §四），
   修掉 `image_max_pixels` 和 `save_steps` 后重跑并跑通 `--tag lora` 对比。
2. **补齐 `docs/Day23~27`**。Day22 当天写了，Day23–26 因为连着跑 GPU 任务没跟上，这里补齐。
   每篇只写**决策理由和踩坑**，实验数据不重复抄——数据在 `deliverables/` 里。
3. **写周报**并转 docx。

## 三、本周没做但值得做的（留给下周）

1. **Gemma 把 `max_soft_tokens` 提到 1120 重跑 Day23 的 OCR 题**。
   本周的归因是「形近字错源于固定 token 预算不足」，但这是**推论，没有做对照实验**。
   提上去如果就对了，归因成立；如果还错，说明是视觉塔容量问题，结论要改。
   **这是本周唯一一条没被实验直接验证的因果判断，必须补。**
2. **幻觉的系统提示缓解**：system prompt 里显式加"图中没有的内容必须回答'图中未显示'"，
   量化抗误导率能提多少。零成本手段，应该先于微调尝试。
3. **Gemma 的 LoRA 对照组**（`configs/gemma4_e4b_lora_sft.yaml` 已写好，超参与主配置完全一致）。
   本周 GPU 时间都花在 Qwen 的两次训练上，没跑。

## 四、归档

```powershell
.\.venv-vlm\Scripts\python.exe Week5\code\md_to_docx.py Week5\deliverables\第5周_多模态实践报告.md
```

`saves/` 与 `models/` 在 `.gitignore` 里（体积大），微调产物不进 git；
`Week5/data/train_images/` 与 `eval_images/` 是脚本可复现生成的（seed 42 / 10042），
删掉也能一条命令重建：

```powershell
.\.venv-vlm\Scripts\python.exe Week5\code\build_vlm_sft_data.py
```
