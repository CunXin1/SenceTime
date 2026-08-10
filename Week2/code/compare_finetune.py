"""Day9：微调前后对比测试。用 3 个新问题对比 [基座] vs [合并后模型] 的回答。

验证验收❹：微调后模型对话质量优于基座。
为省显存，基座与合并模型**依次**加载（跑完一个释放再跑下一个）。

用法（训练+合并完成后）：
    .venv/Scripts/python.exe Week2/code/compare_finetune.py \
        --base models/Qwen2.5-3B-Instruct \
        --merged models/Qwen2.5-3B-week2-merged \
        --out Week2/deliverables/compare_qwen.md
"""
import argparse
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 3 个训练集里没有的新问题（覆盖：概念解释 / 代码 / 建议类）
QUESTIONS = [
    "用一句话解释什么是「过拟合」，并给一个生活中的比喻。",
    "写一个 Python 函数 is_palindrome(s)，判断字符串是否为回文，并简单说明思路。",
    "我想入门机器学习，请推荐三本适合初学者的书，并分别说明推荐理由。",
]


def chat(model_path, questions, max_new_tokens=400):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.eval()
    answers = []
    for q in questions:
        msgs = [{"role": "user", "content": q}]
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        answers.append(ans)
    # 释放显存
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基座模型路径")
    ap.add_argument("--merged", required=True, help="合并后模型路径")
    ap.add_argument("--out", required=True, help="输出 markdown 路径")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    args = ap.parse_args()

    print("=== 加载基座生成 ===")
    base_ans = chat(args.base, QUESTIONS, args.max_new_tokens)
    print("=== 加载合并模型生成 ===")
    ft_ans = chat(args.merged, QUESTIONS, args.max_new_tokens)

    lines = [f"# 微调前后对比：{args.base}  vs  {args.merged}\n",
             "> 3 个训练集外的新问题，贪心解码（do_sample=False）保证可复现。\n"]
    for i, q in enumerate(QUESTIONS):
        lines += [f"## 问题 {i+1}：{q}\n",
                  "### 🔵 基座（微调前）", base_ans[i], "",
                  "### 🟢 微调后（合并模型）", ft_ans[i], "",
                  "---", ""]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"对比已保存 -> {args.out}")


if __name__ == "__main__":
    main()
