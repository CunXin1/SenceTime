"""
quantize_awq.py — Week7 Day34
用 llm-compressor 对 Week4 的 DPO 交付模型做 AWQ W4A16 量化。
AWQ W4A16 quantization of the Week4 DPO model via llm-compressor.

★ AWQ 在做什么（Day34.1 的原理落到代码上）
    LLM 的权重里只有约 1% 的通道是"显著"的（salient），它们对应的**输入激活幅度大**，
    量化误差被激活放大后对输出影响最大。AWQ 的洞见是：与其像 GPTQ 那样逐列补偿误差，
    不如在量化前先给这些显著通道的权重乘一个缩放因子 s（对应输入除以 s，数学等价），
    把它们推到量化格点更密的区间里，量化完再缩回来。
      · 显著性判据用的是**激活幅度**而不是权重幅度 —— 这就是"激活感知"
        (activation-aware) 这个名字的来源，也是它必须要校准集的原因。
      · 缩放因子按**通道分组**搜索（group size 128），不是逐权重，所以额外开销极小。
    与 GPTQ 的分工差异：GPTQ 用 Hessian 做误差补偿（改权重值），AWQ 做等价变换
    （改权重尺度），前者更贵更准，后者更快且对校准集数量不敏感。Day35 对比这两条路线。

★ 为什么用 llm-compressor 而不是 AutoAWQ
    AutoAWQ 仓库已归档停止维护，官方 README 指向 vLLM 的 llm-compressor。
    更实际的理由是产出格式：llm-compressor 出 compressed-tensors，vLLM 原生加载，
    且与 GPTQ 共用同一套 oneshot API，Day35 的对比不会掺进"两个工具实现差异"的噪声。
    Week1 的 week1/code/finetune/export_awq.py 那套 AutoAWQForCausalLM API 仍可作退路。

★ lm_head 为什么不量化
    Qwen2.5-3B 是 tied embedding（输入 embedding 与 lm_head 共享权重，151936×2048
    ≈ 0.31B 参数 / fp16 约 0.62GB）。这一层直接决定 logits 的数值精度，量化它对困惑度
    的伤害远大于省下的那点显存，业界惯例一律 ignore。这也是"3B 模型量化后不是 6.2/4
    而是约 2.2GB"的原因：只有约 2.6B 的线性层参数被压到 4-bit。
    lm_head stays fp16 (tied embeddings, ~0.62GB) — quantizing it hurts PPL badly.

用法 / Usage（★ WSL 的 quant 环境）:
    source ~/venvs/quant/bin/activate
    cd /mnt/c/Users/Ruibo\\'s\\ Desktop/SenceTime_Weeks1-5
    python Week7/code/quantize_awq.py 2>&1 | tee Week7/deliverables/logs/quantize_awq.log
    python Week7/code/quantize_awq.py --check      # 只验证 API 可用，不跑量化
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-3B-week4-dpo-merged"
DEFAULT_OUT = ROOT / "models" / "Qwen2.5-3B-week4-dpo-awq-w4"
CALIB = ROOT / "Week7" / "data" / "calib.json"


def load_api():
    """导入 llm-compressor 的 oneshot 与 AWQ 配方所需的两个 modifier。

    ★ 2026-08-21 实测修正：AWQModifier 在 0.13.0 里被拆成了两个 modifier
        旧写法 `from llmcompressor.modifiers.awq import AWQModifier` 现在是一个
        **废弃 shim**，它返回的不是一个 modifier 而是一个 **list**：
            [AWQModifier(duo_scaling=...), QuantizationModifier(scheme=..., ignore=...)]
        职责被拆开了——前者只负责搜索等价缩放因子（AWQ 的核心变换），后者负责真正
        把权重压到 4-bit。这个拆分是合理的：缩放搜索和量化本来就是两件事，拆开后
        AWQ 的变换可以和别的量化配方组合。
        踩到的坑：shim 返回 list 意味着 `hasattr(recipe, "group_size")` 恒为 False，
        原脚本那句 `--group-size` 赋值**静默失效且不报错**。故改用新 API 显式构造。
        The old single-modifier import is a shim returning a LIST; attribute-probing
        it silently no-ops. Use the explicit two-modifier recipe instead.
    """
    try:
        from llmcompressor import oneshot                      # 新入口
    except ImportError:
        from llmcompressor.transformers import oneshot         # 旧入口
    try:
        from llmcompressor.modifiers.transform.awq import AWQModifier      # >=0.13
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except ImportError as exc:
        raise SystemExit(
            "[FAIL] 找不到 AWQModifier / QuantizationModifier: " + str(exc) + chr(10) +
            "  llm-compressor 版本可能过旧：pip install -U llmcompressor" + chr(10) +
            "  退路：改用 Week1 的 AutoAWQ 方案（week1/code/finetune/export_awq.py）"
        ) from exc
    return oneshot, AWQModifier, QuantizationModifier


def build_recipe(AWQModifier, QuantizationModifier, group_size: int, sym: bool = True):
    """拼出 AWQ W4A16 配方。

    ★ group_size 不是 modifier 的参数，它在 **scheme** 里
        `preset_name_to_scheme("W4A16", ["Linear"])` 出来的就是
        num_bits=4 / group_size=128 / symmetric=True / strategy=group——
        也就是说 128 正是预设值，绝大多数情况直接用预设名即可。
        只有要偏离 128 时才需要把预设展开成 config_groups 再改字段。
        group_size 决定「多少个权重共享一组 scale/zero-point」：越小越准、
        但每组的 scale 本身也要存 fp16，128 是精度与额外开销的业界折中。
    """
    preset = "W4A16" if sym else "W4A16_ASYM"
    qm_kwargs = {"targets": ["Linear"], "ignore": ["lm_head"]}   # lm_head 见文件头
    if group_size == 128:
        qm_kwargs["scheme"] = preset
    else:
        from compressed_tensors.quantization import preset_name_to_scheme
        scheme = preset_name_to_scheme(preset, ["Linear"])
        scheme.weights.group_size = group_size
        qm_kwargs["config_groups"] = {"group_0": scheme}
    # AWQModifier 只做等价缩放搜索；真正的 4-bit 压缩由 QuantizationModifier 完成。
    return [AWQModifier(), QuantizationModifier(**qm_kwargs)]


def build_dataset(model_path: Path, maxlen: int, nsamples: int):
    """把 calib.json 预分词成 input_ids，直接交给 oneshot。

    ★ 为什么自己分词而不是让 oneshot 去处理原始文本
        oneshot 支持传数据集名字让它内部走预设的 preprocessing，但那套预设绑定了
        它自己的 chat 模板约定。我们的 calib.json 已经是按 Qwen ChatML 拼好的长文档
        （见 build_calib_data.py），再过一遍它的模板会套两层 <|im_start|>。
        预分词后传 Dataset 是它文档里的 custom dataset 路径，最可控。
    """
    from datasets import Dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    docs = json.loads(CALIB.read_text(encoding="utf-8"))[:nsamples]
    rows = [tok(d["text"], max_length=maxlen, truncation=True) for d in docs]
    ds = Dataset.from_list([{"input_ids": r["input_ids"],
                             "attention_mask": r["attention_mask"]} for r in rows])
    print(f"[calib] {len(ds)} 条样本, maxlen={maxlen}")
    return ds


def dir_size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 ** 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--maxlen", type=int, default=1024)
    # ★ AWQ 的校准样本数可以比 GPTQ 少：它只需要估计每个通道的激活幅度均值，
    #   统计量比 GPTQ 的 Hessian 稳定得多。但这里仍用 128 与 GPTQ 对齐，
    #   让 Day35 的对比只剩"算法差异"这一个变量。
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--check", action="store_true", help="只验证依赖可用，不跑量化")
    args = ap.parse_args()

    oneshot, AWQModifier, QuantizationModifier = load_api()
    import llmcompressor
    print(f"[env] llmcompressor={getattr(llmcompressor, '__version__', '?')}")
    if args.check:
        print("[ok] API 可用（--check 模式，未执行量化）")
        return

    if not args.model.exists():
        raise SystemExit(f"[FAIL] 输入模型不存在: {args.model}")
    if not CALIB.exists():
        raise SystemExit(f"[FAIL] 校准集不存在，先在 Windows 侧跑 build_calib_data.py: {CALIB}")
    if args.out.exists():
        print(f"[warn] 输出目录已存在，将被覆盖: {args.out}")
        shutil.rmtree(args.out)

    ds = build_dataset(args.model, args.maxlen, args.nsamples)

    # W4A16 = 权重 4-bit、激活保持 16-bit。不做 A8/A4 的理由：激活量化要在推理时
    # 动态处理 outlier，收益主要体现在超大 batch 的算力瓶颈场景；4090 单卡小 batch
    # 是**显存带宽**瓶颈，压权重就够了，压激活只会白白多一份精度损失。
    recipe = build_recipe(AWQModifier, QuantizationModifier, args.group_size)
    print(f"[recipe] {[type(m).__name__ for m in recipe]}  group_size={args.group_size}")

    t0 = time.time()
    oneshot(
        model=str(args.model),
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.maxlen,
        num_calibration_samples=len(ds),
        output_dir=str(args.out),
    )
    mins = (time.time() - t0) / 60

    src_gb, out_gb = dir_size_gb(args.model), dir_size_gb(args.out)
    print(f"\n[ok] AWQ 完成 -> {args.out}")
    print(f"[ok] 耗时 {mins:.1f} 分钟；磁盘 {src_gb:.2f} GB -> {out_gb:.2f} GB "
          f"(降 {100 * (1 - out_gb / src_gb):.1f}%)")
    print("[note] 磁盘体积只是参考，验收要的『显存降低』以 Week7/code/bench_quant.py "
          "读 vLLM 启动日志的权重占用为准。")


if __name__ == "__main__":
    main()
