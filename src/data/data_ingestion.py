from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

# project root = 3 levels up from this file (src/data/data_ingestion.py)
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw"

# ---- settings (change these) ----
DATA_URL = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
TEXT_COL = "clean_comment"
TARGET_COL = "category"
params = yaml.safe_load(open(ROOT /"params.yaml"))["data_ingestion"]
TEST_SIZE = params["test_size"]
RANDOM_STATE = params["random_state"]


def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    print("loaded:", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=[TEXT_COL, TARGET_COL])             # 1. missing values
    df = df[df[TEXT_COL].astype(str).str.strip() != ""]       # 2. empty strings
    df = df.drop_duplicates().reset_index(drop=True)          # 3. duplicates
    print("after cleaning:", df.shape)
    return df


def split_and_save(df: pd.DataFrame) -> None:
    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],   # keeps class ratio same in both
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    print("saved:", train.shape, test.shape, "->", OUT_DIR)


def main() -> None:
    df = load_data(DATA_URL)
    df = clean_data(df)
    split_and_save(df)


if __name__ == "__main__":
    main()