"""Error analysis of the stacking model on the test set.

Writes reports/error_analysis.md
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

LABELS = [-1, 0, 1]
NAME = {-1: "negative", 0: "neutral", 1: "positive"}


def short(text, n=220):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "..."


def load_texts(y_test):
    """Original comments if they line up with the test rows, else cleaned text."""
    raw = pd.read_csv(DATA / "raw" / "test.csv")
    clean = pd.read_csv(DATA / "interim" / "test_processed.csv")
    text_col = "clean_comment" if "clean_comment" in raw else raw.select_dtypes("object").columns[0]
    if "category" in raw and len(raw) == len(y_test) and (raw["category"].to_numpy() == y_test).all():
        return raw[text_col].astype(str).tolist(), "original text"
    print("NOTE: raw test.csv does not line up with X_test; using the cleaned text instead")
    return clean["clean_comment"].astype(str).tolist(), "cleaned text (stopwords removed)"


def main():
    X_test = sparse.load_npz(DATA / "processed" / "X_test.npz")
    y_test = np.load(DATA / "processed" / "y_test.npy")
    with open(MODELS / "stacking_model.pkl", "rb") as f:
        stack = pickle.load(f)
    with open(MODELS / "tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    texts, text_kind = load_texts(y_test)
    proba = stack.predict_proba(X_test)
    pred = stack.classes_[proba.argmax(axis=1)]

    df = pd.DataFrame({"text": texts, "true": y_test, "pred": pred, "conf": proba.max(axis=1)})
    df["words"] = df["text"].str.split().str.len()
    df["wrong"] = df["true"] != df["pred"]
    wrong = df[df["wrong"]]

    out = ["# Error analysis", "",
           f"Test comments: {len(df)}. Mistakes: {len(wrong)} "
           f"(accuracy {1 - len(wrong) / len(df):.4f}). Examples show the {text_kind}.", ""]

    # 1) Confusion matrix, as counts and % of each true class
    cm = confusion_matrix(df["true"], df["pred"], labels=LABELS)
    out += ["## Confusion matrix (rows = true, columns = predicted)", "",
            "| true \\ predicted | negative | neutral | positive |", "|---|---|---|---|"]
    for i, lab in enumerate(LABELS):
        cells = [f"{n} ({n / cm[i].sum():.0%})" for n in cm[i]]
        out.append(f"| {NAME[lab]} | " + " | ".join(cells) + " |")
    out.append("")

    # 2) Which mistakes are most common
    pairs = wrong.groupby(["true", "pred"]).size().sort_values(ascending=False)
    out += ["## Most common mistakes", ""]
    for (t, p), n in pairs.items():
        out.append(f"- true {NAME[t]}, predicted {NAME[p]}: {n}")
    out.append("")

    # 3) Accuracy by comment length
    df["length"] = pd.cut(df["words"], [-1, 5, 15, 40, 100000],
                          labels=["1-5 words", "6-15 words", "16-40 words", "41+ words"])
    out += ["## Accuracy by comment length", "", "| length | comments | accuracy |", "|---|---|---|"]
    for name, g in df.groupby("length", observed=True):
        out.append(f"| {name} | {len(g)} | {1 - g['wrong'].mean():.3f} |")
    out.append("")

    # 4) Accuracy by model confidence
    df["confidence"] = pd.cut(df["conf"], [0, 0.5, 0.7, 0.9, 1.0001],
                              labels=["<0.5", "0.5-0.7", "0.7-0.9", "0.9+"])
    out += ["## Accuracy by model confidence", "", "| confidence | comments | accuracy |", "|---|---|---|"]
    for name, g in df.groupby("confidence", observed=True):
        out.append(f"| {name} | {len(g)} | {1 - g['wrong'].mean():.3f} |")
    out.append("")

    # 5) Random examples of the top 3 mistake types
    out += ["## Example mistakes", ""]
    for (t, p), _ in list(pairs.items())[:3]:
        out.append(f"### True {NAME[t]}, predicted {NAME[p]}")
        sample = wrong[(wrong["true"] == t) & (wrong["pred"] == p)].sample(
            min(8, ((wrong["true"] == t) & (wrong["pred"] == p)).sum()), random_state=42)
        for _, r in sample.iterrows():
            out.append(f'- "{short(r["text"])}" (confidence {r["conf"]:.2f})')
        out.append("")

    # 6) Confident mistakes: often label noise or real model blind spots
    out += ["## Most confident mistakes", ""]
    for _, r in wrong.sort_values("conf", ascending=False).head(10).iterrows():
        out.append(f'- true {NAME[r["true"]]}, predicted {NAME[r["pred"]]} '
                   f'({r["conf"]:.2f}): "{short(r["text"])}"')
    out.append("")

    # 7) Words each class relies on (logistic regression base model)
    lr = stack.named_estimators_["lr"]
    vocab = np.array(tfidf.get_feature_names_out())
    out += ["## Top words per class (logistic regression coefficients)", ""]
    # lr.classes_ are the internal codes 0,1,2; stack.classes_ are the real labels -1,0,1
    for i, cls in enumerate(stack.classes_):
        top = vocab[np.argsort(lr.coef_[i])[-15:][::-1]]
        out.append(f"- **{NAME[int(cls)]}**: " + ", ".join(top))
    out.append("")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "error_analysis.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {REPORTS / 'error_analysis.md'}")
    print(pairs.to_string())


if __name__ == "__main__":
    main()