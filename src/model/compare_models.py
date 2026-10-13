"""Compare each base model on its own against the stacking ensemble.

Same features, same test set and same hyperparameters as the pipeline.
Writes reports/model_comparison.csv and reports/model_comparison.md
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.model.model_building import build_model  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "stacking_model.pkl"
REPORTS = ROOT / "reports"

LABELS = [-1, 0, 1]  # negative, neutral, positive
NAMES = {
    "lgbm": "LightGBM",
    "svc": "Linear SVM (calibrated)",
    "lr": "Logistic Regression",
    "nb": "Complement Naive Bayes",
}


def score(name, y_true, y_pred, train_seconds):
    """One row of the table: overall and per-class metrics."""
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None)
    return {
        "model": name,
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_negative": round(per_class[0], 4),
        "f1_neutral": round(per_class[1], 4),
        "f1_positive": round(per_class[2], 4),
        "train_seconds": train_seconds,
    }


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = ["-" if pd.isna(v) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    X_train = sparse.load_npz(DATA_DIR / "X_train.npz")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_test = sparse.load_npz(DATA_DIR / "X_test.npz")
    y_test = np.load(DATA_DIR / "y_test.npy")

    rows = []

    # 1) Floor: always predict the most common class
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    rows.append(score("Majority-class baseline", y_test, dummy.predict(X_test), None))

    # 2) Each base model on its own, with the pipeline's exact settings
    for key, est in build_model().estimators:
        model = clone(est)
        print(f"training {NAMES[key]} alone...")
        start = time.time()
        model.fit(X_train, y_train)
        seconds = round(time.time() - start, 1)
        rows.append(score(NAMES[key], y_test, model.predict(X_test), seconds))

    # 3) The saved stacking model (not retrained, so it matches the README)
    with open(MODEL_PATH, "rb") as f:
        stack = pickle.load(f)
    rows.append(score("Stacking ensemble (final)", y_test, stack.predict(X_test), None))

    df = pd.DataFrame(rows)

    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "model_comparison.csv", index=False)
    (REPORTS / "model_comparison.md").write_text(to_markdown(df), encoding="utf-8")

    print()
    print(to_markdown(df))

    singles = df[df["model"].isin(NAMES.values())]
    best = singles.loc[singles["macro_f1"].idxmax()]
    final = df[df["model"] == "Stacking ensemble (final)"].iloc[0]
    print(f"\nBest single model: {best['model']} (macro F1 {best['macro_f1']})")
    print(f"Stack vs best single: {final['macro_f1'] - best['macro_f1']:+.4f} macro F1")


if __name__ == "__main__":
    main()