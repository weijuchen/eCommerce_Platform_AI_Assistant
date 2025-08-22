import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType


# 1. 讀取 FAQ 資料
df = pd.read_csv("../data/faq.csv")

# 確保欄位名稱正確
if not {"question", "answer"}.issubset(df.columns):
    raise ValueError("CSV 必須包含 'question' 和 'answer' 欄位")

faq_texts = [f"Q: {q} A: {a}" for q, a in zip(df["question"], df["answer"])]

# 2. 建立 HuggingFace Dataset
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 沒有 pad_token，要補上

dataset = Dataset.from_dict({"text": faq_texts})


def tokenize_fn(batch):
    encoding = tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=128
    )
    encoding["labels"] = encoding["input_ids"].copy()  # labels 必須加上，才能算 loss
    return encoding


dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# 3. 載入模型 + 加上 LoRA
model = AutoModelForCausalLM.from_pretrained("gpt2")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.1  # 因為是語言模型
)

model = get_peft_model(model, lora_config)

# 4. 設定訓練參數
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    # evaluation_strategy="no",
    fp16=False,  # 如果有 GPU with FP16 可以設 True
    report_to="none",
)

# 5. 建立 Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

# 6. 開始訓練
trainer.train()

# 7. 儲存 LoRA 權重
model.save_pretrained("./outputs/lora_faq")
tokenizer.save_pretrained("./outputs/lora_faq")

print("✅ LoRA 微調完成，權重已儲存於 ./outputs/lora_faq")
