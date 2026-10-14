"""Domain-shift check: score the model on hand-labelled YouTube comments.

Reads data/external/youtube_sample.csv (columns: video_id, comment, label),
predicts with the SAME cleaning + model files the API uses, and writes
reports/domain_shift.md. Run from the project root:

    python src/model/domain_shift.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# importing from the backend guarantees identical cleaning and model files
from backend.app import clean_text, model, vectorizer  # noqa: E402

SAMPLE = ROOT / "data" / "external" / "youtube_sample.csv"
REPORTS = ROOT / "reports"
LABELS = [-1, 0, 1]
NAME = {-1: "negative", 0: "neutral", 1: "positive"}


def short(text, n=200):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "..."


def scores(y_true, y_pred):
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def main():
    df = pd.read_csv(SAMPLE, encoding="utf-8-sig")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    total = len(df)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    if df.empty:
        sys.exit("No labelled rows found. Fill the label column with -1 / 0 / 1 first.")

    df["clean"] = df["comment"].astype(str).map(clean_text)
    proba = model.predict_proba(vectorizer.transform(df["clean"]))
    df["pred"] = model.classes_[proba.argmax(axis=1)]
    df["conf"] = proba.max(axis=1)
    df["wrong"] = df["label"] != df["pred"]
    df["empty_after_clean"] = df["clean"].str.strip() == ""

    acc, mf1 = scores(df["label"], df["pred"])
    majority = df["label"].value_counts(normalize=True).max()

    out = ["# Domain-shift check", "",
           "The model is trained on Reddit comments. Here it is scored on YouTube comments "
           "that I labelled by hand (all from Hinglish-language videos).", "",
           f"Labelled comments scored: **{len(df)}** (of {total} sampled; blanks skipped).", ""]

    # 1) Headline numbers
    out += ["## Headline", "", "| | value |", "|---|---|",
            f"| Accuracy | {acc:.3f} |",
            f"| Macro F1 | {mf1:.3f} |",
            f"| Majority-class accuracy (baseline) | {majority:.3f} |",
            "| Reddit test macro F1 (for comparison) | 0.864 |", ""]

    # 2) My labels vs the model's predictions
    out += ["## Label distribution", "", "| class | my labels | model predictions |", "|---|---|---|"]
    for lab in LABELS:
        out.append(f"| {NAME[lab]} | {(df['label'] == lab).sum()} | {(df['pred'] == lab).sum()} |")
    out.append("")

    # 3) Per-class F1
    f1s = f1_score(df["label"], df["pred"], labels=LABELS, average=None, zero_division=0)
    out += ["## Per-class F1", "", "| class | F1 | support |", "|---|---|---|"]
    for lab, f in zip(LABELS, f1s):
        out.append(f"| {NAME[lab]} | {f:.3f} | {(df['label'] == lab).sum()} |")
    out.append("")

    # 4) Confusion matrix
    cm = confusion_matrix(df["label"], df["pred"], labels=LABELS)
    out += ["## Confusion matrix (rows = my label, columns = predicted)", "",
            "| mine \\ predicted | negative | neutral | positive |", "|---|---|---|---|"]
    for i, lab in enumerate(LABELS):
        out.append(f"| {NAME[lab]} | " + " | ".join(str(n) for n in cm[i]) + " |")
    out.append("")

    # 5) Per video
    out += ["## Per video", "", "| video | comments | accuracy | macro F1 |", "|---|---|---|---|"]
    for vid, g in df.groupby("video_id"):
        a, f = scores(g["label"], g["pred"])
        out.append(f"| {vid} | {len(g)} | {a:.3f} | {f:.3f} |")
    out.append("")

    # 6) Cleaning wipes out non-Latin text (the API's cleaner keeps only A-Z, 0-9, basic punctuation)
    empty = df[df["empty_after_clean"]]
    out += ["## Comments with no usable text after cleaning", "",
            f"{len(empty)} of {len(df)} comments are empty after cleaning "
            "(emoji-only, or written in Devanagari / other non-Latin script). "
            "The model can only guess from the bias term for these.", ""]
    if len(empty):
        a_empty = 1 - empty["wrong"].mean()
        out.append(f"Accuracy on those: {a_empty:.3f}. Accuracy on the rest: "
                   f"{1 - df[~df['empty_after_clean']]['wrong'].mean():.3f}.")
        out.append("")

    # 7) Accuracy by confidence
    df["confidence"] = pd.cut(df["conf"], [0, 0.5, 0.7, 0.9, 1.0001],
                              labels=["<0.5", "0.5-0.7", "0.7-0.9", "0.9+"])
    out += ["## Accuracy by model confidence", "", "| confidence | comments | accuracy |", "|---|---|---|"]
    for name, g in df.groupby("confidence", observed=True):
        out.append(f"| {name} | {len(g)} | {1 - g['wrong'].mean():.3f} |")
    out.append("")

    # 8) Most confident mistakes
    out += ["## Most confident mistakes", ""]
    for _, r in df[df["wrong"]].sort_values("conf", ascending=False).head(10).iterrows():
        out.append(f'- mine {NAME[r["label"]]}, predicted {NAME[r["pred"]]} '
                   f'({r["conf"]:.2f}): "{short(r["comment"])}"')
    out.append("")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "domain_shift.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {REPORTS / 'domain_shift.md'}")
    print(f"n={len(df)}  accuracy={acc:.3f}  macro F1={mf1:.3f}  majority baseline={majority:.3f}")
    print(pd.crosstab(df["label"], df["pred"]).to_string())


if __name__ == "__main__":
    main()