import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "data" / "interim"
OUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

TEXT_COL = "clean_comment"
TARGET_COL = "category"

params = yaml.safe_load(open(ROOT / "params.yaml"))["feature_engineering"]
MAX_FEATURES = params["max_features"]
NGRAM_RANGE = tuple(params["ngram_range"])


def main() -> None:
    train = pd.read_csv(IN_DIR / "train_processed.csv").dropna(subset=[TEXT_COL])
    test = pd.read_csv(IN_DIR / "test_processed.csv").dropna(subset=[TEXT_COL])

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    X_train = vectorizer.fit_transform(train[TEXT_COL])   # learn vocabulary from TRAIN only
    X_test = vectorizer.transform(test[TEXT_COL])         # just apply it to test

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(OUT_DIR / "X_train.npz", X_train)
    sparse.save_npz(OUT_DIR / "X_test.npz", X_test)
    np.save(OUT_DIR / "y_train.npy", train[TARGET_COL].to_numpy())
    np.save(OUT_DIR / "y_test.npy", test[TARGET_COL].to_numpy())
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("X_train:", X_train.shape, "X_test:", X_test.shape)


if __name__ == "__main__":
    main()