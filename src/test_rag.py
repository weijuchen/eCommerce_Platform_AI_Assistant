import pandas as pd

df = pd.read_csv("../data/FAQ.csv", encoding="utf-8-sig")
# df = pd.read_csv("faq.csv", encoding="utf-8-sig")  # 建議用 utf-8-sig

print("欄位名稱：", df.columns.tolist())
print(df.head())
