import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "interim"

TEXT_COL = "clean_comment"
KEEP_WORDS = {"not", "but", "however", "no", "yet"}   # sentiment words, don't remove

STOP_WORDS = set(stopwords.words("english")) - KEEP_WORDS
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("\n", " ")
    text = re.sub(r"[^A-Za-z0-9\s!?.,]", "", text)             # remove special characters
    words = [w for w in text.split() if w not in STOP_WORDS]   # remove stopwords
    words = [LEMMATIZER.lemmatize(w) for w in words]           # lemmatize
    return " ".join(words)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
    df = df[df[TEXT_COL].str.strip() != ""]    # rows that became empty
    return df.reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["train", "test"]:
        df = pd.read_csv(IN_DIR / f"{name}.csv")
        out = preprocess(df)
        out.to_csv(OUT_DIR / f"{name}_processed.csv", index=False)
        print(f"{name}: {df.shape} -> {out.shape}")


if __name__ == "__main__":
    main()