import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def main():
    train_df = pd.read_csv("msr_paraphrase_train.txt", sep="\t", quoting=3, on_bad_lines="skip")
    test_df = pd.read_csv("msr_paraphrase_test.txt", sep="\t", quoting=3, on_bad_lines="skip")

    train_df = train_df[["Quality", "#1 String", "#2 String"]].dropna().reset_index(drop=True)
    test_df = test_df[["Quality", "#1 String", "#2 String"]].dropna().reset_index(drop=True)

    for df in (train_df, test_df):
        df["Quality"] = df["Quality"].astype(int)
        df["s1_clean"] = df["#1 String"].apply(clean_text)
        df["s2_clean"] = df["#2 String"].apply(clean_text)

    train_split, val_split = train_test_split(
        train_df,
        test_size=0.15,
        random_state=42,
        stratify=train_df["Quality"]
    )

    train_split[["Quality", "s1_clean", "s2_clean"]].to_csv("train_clean.csv", index=False)
    val_split[["Quality", "s1_clean", "s2_clean"]].to_csv("val_clean.csv", index=False)
    test_df[["Quality", "s1_clean", "s2_clean"]].to_csv("test_clean.csv", index=False)

if __name__ == "__main__":
    main()