"""Gemma-4-E2B-it 推理(多模态模型)。

Gemma-4 与 Qwen/Llama 不同,不能只用 AutoTokenizer,要用官方方式:
    - AutoProcessor(内含 tokenizer + 多模态处理 + 正确的聊天模板)
    - processor.apply_chat_template(..., enable_thinking=False) 关思考
    - processor.parse_response() 提取干净回答(自动处理 EOS/特殊符号)
参考:HuggingFace 官方 model card (google/gemma-4-E2B-it)。

用法(在 Week1/ 根目录 + llm_exp 环境运行):
    conda activate llm_exp && cd ~/SenceTime/Week1
    python code/inference/inference_gemma4.py
"""
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL = "./models/gemma-4-E2B-it"
DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available()
          else "cpu")

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map=DEVICE).eval()


def chat(user_msg):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_msg},
    ]
    # 官方模板;enable_thinking=False 关闭思考模式,输出更干净
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=text, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(  # Gemma 官方推荐采样参数
            **inputs, max_new_tokens=512,
            do_sample=True, temperature=1.0, top_p=0.95, top_k=64)

    # 只取新生成部分,交给 processor 解析出干净回答
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    try:
        parsed = processor.parse_response(response)  # 去掉特殊符号/思考块
        # parse_response 返回 {'role':..., 'content':...},取正文
        return parsed["content"] if isinstance(parsed, dict) else parsed
    except Exception:
        return processor.decode(outputs[0][input_len:], skip_special_tokens=True)


TESTS = [
    ("代码生成", "用 Python 写一个判断字符串是否回文的函数,并给测试例子。"),
    ("逻辑推理", "小明比小红大3岁,小红比小刚大2岁,小刚10岁。小明几岁?一步步推理。"),
    ("角色扮演", "你是中学物理老师,用生活例子给初中生解释牛顿第一定律。"),
]

if __name__ == "__main__":
    print(f"[加载] {MODEL} @ {DEVICE}")
    for domain, prompt in TESTS:
        print(f"\n{'='*60}\n【{domain}】{prompt}\n{'-'*60}")
        print(chat(prompt))
