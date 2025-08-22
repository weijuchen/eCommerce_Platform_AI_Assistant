# 使用lora 且自動更新

import os
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# ================= 0. 參數設定 =================
FAQ_CSV_PATH = "../data/FAQ.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_MODEL_NAME = "uer/gpt2-chinese-cluecorpussmall"
LORA_MODEL_PATH = "./train_lora"
VECTORSTORE_PATH = "./faiss_faq.index"  # FAISS 向量庫存放路徑


# ================= 1. 載入 FAQ =================
def load_faq(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    texts = [f"Q: {q} A: {a}" for q, a in zip(df["question"], df["answer"])]
    return texts


faq_texts = load_faq(FAQ_CSV_PATH)

# ================= 2. 建立或更新向量資料庫 =================
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

if os.path.exists(VECTORSTORE_PATH):
    # 載入已有向量庫
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings)
    # 檢查是否需要更新
    existing_count = len(vectorstore.index_to_text)
    if existing_count != len(faq_texts):
        vectorstore = FAISS.from_texts(faq_texts, embedding=embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)
else:
    vectorstore = FAISS.from_texts(faq_texts, embedding=embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

# ================= 3. 載入微調後 LoRA 模型 =================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)

# 載入 LoRA 微調權重
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=False,
)

llm = HuggingFacePipeline(pipeline=pipe)

# ================= 4. 自訂 Prompt =================
custom_prompt = PromptTemplate(
    template="""
你是一個購物平台的智慧客服，以下是可能有幫助的 FAQ：

{context}

根據以上 FAQ，請用中文回答客戶的問題。
如果 FAQ 裡沒有相關答案，請回覆「抱歉，這個問題目前沒有資訊」。

問題: {question}
回答:""",
    input_variables=["context", "question"],
)

# ================= 5. 建立 RAG QA Chain =================
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
)


# ================= 6. 提問函數 =================
def ask_question(question):
    result = qa.invoke({"query": question})
    return result["result"]


# ================= 7. 測試 =================
if __name__ == "__main__":
    print("==== 測試 ====")
    questions = ["我要怎麼退貨？", "運費是多少？", "有提供國際運送嗎？"]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {ask_question(q)}\n")


# 使用lora 但沒有自動更新

# import pandas as pd
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFacePipeline

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from peft import PeftModel

# # ================= 1. 載入 FAQ =================
# df = pd.read_csv("../data/FAQ.csv", encoding="utf-8")
# faq_texts = [f"Q: {q} A: {a}" for q, a in zip(df["question"], df["answer"])]

# # ================= 2. 建立向量資料庫 =================
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = FAISS.from_texts(faq_texts, embedding=embeddings)

# # ================= 3. 載入微調後 LoRA 中文 GPT2 模型 =================
# base_model_name = "uer/gpt2-chinese-cluecorpussmall"
# lora_model_path = "./faq_lora_model"  # 之前微調後 LoRA 模型的存放路徑

# tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# # 載入 LoRA 微調權重
# model = PeftModel.from_pretrained(base_model, lora_model_path)

# # 建立生成管線
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=100,
#     do_sample=False,
# )

# llm = HuggingFacePipeline(pipeline=pipe)

# # ================= 4. 自訂 Prompt =================
# custom_prompt = PromptTemplate(
#     template="""
# 你是一個購物平台的智慧客服，以下是可能有幫助的 FAQ：

# {context}

# 根據以上 FAQ，請用中文回答客戶的問題。
# 如果 FAQ 裡沒有相關答案，請回覆「抱歉，這個問題目前沒有資訊」。

# 問題: {question}
# 回答:""",
#     input_variables=["context", "question"],
# )

# # ================= 5. 建立 RAG QA Chain =================
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=vectorstore.as_retriever(),
#     chain_type="stuff",
#     chain_type_kwargs={"prompt": custom_prompt},
# )


# # ================= 6. 提問測試 =================
# def ask_question(question):
#     result = qa.invoke({"query": question})
#     return result["result"]


# if __name__ == "__main__":
#     print("==== 測試 ====")
#     print(ask_question("我要怎麼退貨？"))
#     print(ask_question("運費是多少？"))
#     print(ask_question("有提供國際運送嗎？"))
