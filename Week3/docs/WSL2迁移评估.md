# WSL2 迁移评估（Week4 开工前决策）

> 背景：Week3 实测 Windows 原生训练的两个痛点——①桌面程序抢 GPU 时间片导致训练
> 无故慢 65%（已靠"清场"缓解）；②加速生态（Unsloth/FlashAttention/Triton）在
> Windows 都是社区轮子，版本组合受限。本文评估是否在 Week4（DPO）前迁到 WSL2。

## 预期收益

| 项 | 幅度 | 依据 |
|---|---|---|
| 摆脱 WDDM 显示驱动层 | 训练快 5~15% | NVIDIA 官方 WSL2 CUDA 文档与社区实测 |
| 加速库全部官方支持 | FA2/Unsloth/Liger/torch.compile 免折腾 | 均只官方发布 Linux 包 |
| OpenCompass 官方支持环境 | 排障成本大降 | 官方只支持 Linux |
| vLLM 可用 | Week4+ 推理/采样提速（DPO 造偏好数据用得上） | vLLM 无 Windows 原生版 |

## 成本与风险

| 项 | 说明 |
|---|---|
| 迁移工时 | 约半天：装 WSL2 + Ubuntu → CUDA on WSL → 重建 venv → 重装 LLaMA-Factory |
| 磁盘 | 模型/数据可跨系统共享（/mnt/c 访问 NTFS），但**跨文件系统 IO 慢**， 建议把 models/、data/ 复制进 WSL ext4（约 +20GB） |
| VRAM 开销 | GPU-PV 虚拟化有少量显存/延迟开销（对我们 13GB 峰值无压力） |
| 断点 | Week3 产物（adapter/日志）留在 Windows 侧即可，无需迁移 |

## 建议

- **Week4 开工第一件事做迁移**（半天投入，此后每周都在赚）：DPO 需要造偏好数据
 （大量推理采样，vLLM 收益大）+ 继续训练（加速库收益稳定）。
- 迁移后首个动作：用 Week3 的冒烟配置对比 WSL2 vs Windows 的 s/step，把数据记录
  进 Week4 文档（延续本周的"先测量再决策"方法论）。

## 迁移步骤清单（半天）

```powershell
# ① Windows 侧（管理员）
wsl --install -d Ubuntu-24.04        # 装 WSL2 + Ubuntu
# ② 驱动：Windows 侧 NVIDIA 驱动已含 WSL 支持，无需在 Linux 内装驱动
```

```bash
# ③ Ubuntu 内
sudo apt update && sudo apt install -y python3.12-venv git
python3.12 -m venv ~/venv-lf && source ~/venv-lf/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/hiyouga/LLaMA-Factory && cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install flash-attn --no-build-isolation liger-kernel   # Linux 官方包
# ④ 数据与模型复制进 ext4（避免 /mnt/c 慢 IO）
cp -r /mnt/c/Users/.../SenceTime_Week1/{models,Week2/data} ~/SenceTime/
# ⑤ 冒烟对比 s/step，记录进 Week4 文档
```
