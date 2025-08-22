import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import evaluate

# 1. 載入 FAQ 資料
df = pd.read_csv("../data/faq.csv")
faq_pairs = list(zip(df["question"], df["answer"]))

# 2. 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 原始 GPT2
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
base_pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer)

# 微調後 GPT2 + LoRA
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = PeftModel.from_pretrained(model, "../outputs/lora_faq")
lora_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 3. 設定評估工具
rouge = evaluate.load("rouge")


# 4. 測試函數
def generate_answer(pipe, question):
    prompt = f"Q: {question} A:"
    output = pipe(prompt, max_new_tokens=50, do_sample=True, top_k=50)[0][
        "generated_text"
    ]
    return output.split("A:")[-1].strip()


# 5. 隨機挑 5 題來測試
sample_data = random.sample(faq_pairs, min(5, len(faq_pairs)))

base_preds, lora_preds, refs = [], [], []

print("=== 測試開始 ===\n")

for q, ref in sample_data:
    base_ans = generate_answer(base_pipe, q)
    lora_ans = generate_answer(lora_pipe, q)

    base_preds.append(base_ans)
    lora_preds.append(lora_ans)
    refs.append(ref)

    print(f"問題: {q}")
    print(f"標準答案: {ref}")
    print(f"原始 GPT2 回答: {base_ans}")
    print(f"微調後 GPT2+LoRA 回答: {lora_ans}")
    print("-" * 60)

# 6. 計算 ROUGE
print("\n=== 評估結果 ===")
print("原始 GPT2:", rouge.compute(predictions=base_preds, references=refs))
print("微調後 GPT2+LoRA:", rouge.compute(predictions=lora_preds, references=refs))
