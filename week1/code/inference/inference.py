"""Qwen2.5-1.5B-Instruct 原生推理 (Day 2)

理解对话背后的 4 步管线:
    ① tokenize        文字 → token id
    ② chat_template   套上 ChatML 对话骨架 (<|im_start|>...<|im_end|>)
    ③ generate        自回归逐 token 生成
    ④ decode          token id → 文字

用法(在 Week1/ 根目录 + llm_exp 环境运行):
    conda activate llm_exp && cd ~/SenceTime/Week1
    python code/inference/inference.py [模型路径]              # 默认 Qwen2.5-1.5B
    python code/inference/inference.py [模型路径] --template   # 额外打印对话模板原文
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 第一个非 --开头的参数当模型路径,缺省用 Qwen2.5
args = [a for a in sys.argv[1:] if not a.startswith("--")]
MODEL = args[0] if args else "./models/Qwen2.5-1.5B-Instruct"

# 设备无关:Mac 用 mps,4090 用 cuda,都没有则退回 cpu(同代码只改这一处)
DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available()
          else "cpu")


def load():
    """加载分词器和模型。fp16 半精度,1.5B 约占 3GB 显存。"""
    print(f"[加载] {MODEL} @ {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=DEVICE)
    model.eval()  # 推理模式:关掉 dropout 等训练期行为
    return tokenizer, model


def chat(tokenizer, model, user_msg, show_template=False):
    """把一句用户输入走完 4 步管线,返回模型回答。"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_msg},
    ]

    # ② 套 ChatML 骨架;add_generation_prompt 在末尾补 "assistant" 引导模型作答
    # enable_thinking=False 关掉 Qwen3 的思考模式;不支持该参数的模型会走 except 分支
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    if show_template:
        print(f"--- ChatML 输入(模型真正吃进去的) ---\n{text}\n{'-'*40}")

    # ① 文字 → token id,并搬到目标设备
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    # ③ 自回归生成(do_sample+temperature 控制随机性)
    with torch.no_grad():  # 推理不需要梯度,省内存、加速
        output = model.generate(
            **inputs, max_new_tokens=512,
            do_sample=True, temperature=0.7, top_p=0.8)

    # ④ 去掉输入那段,只解码新生成的 token
    new_tokens = output[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# Day 2.3:三个领域的测试 prompt
TESTS = [
    ("代码生成", "用 Python 写一个判断字符串是否回文的函数,并给测试例子。"),
    ("逻辑推理", "小明比小红大3岁,小红比小刚大2岁,小刚10岁。小明几岁?一步步推理。"),
    ("角色扮演", "你是中学物理老师,用生活例子给初中生解释牛顿第一定律。"),
]


def main():
    show_template = "--template" in sys.argv
    tokenizer, model = load()
    for domain, prompt in TESTS:
        print(f"\n{'='*60}\n【{domain}】{prompt}\n{'-'*60}")
        answer = chat(tokenizer, model, prompt, show_template)
        print(answer)


if __name__ == "__main__":
    main()
