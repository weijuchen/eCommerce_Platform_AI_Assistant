# update_faq.py
import pandas as pd


def update_faq(new_faq_path="../data/new_faq.csv", faq_path="../data/faq.csv"):
    # 讀取舊 FAQ
    old_df = pd.read_csv(faq_path)

    # 讀取新 FAQ
    new_df = pd.read_csv(new_faq_path)

    # 合併 + 去重複
    merged_df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates()

    # 更新存檔
    merged_df.to_csv(faq_path, index=False, encoding="utf-8-sig")
    print(f"✅ FAQ 更新完成，共 {len(merged_df)} 條 Q&A。")


if __name__ == "__main__":
    update_faq()
