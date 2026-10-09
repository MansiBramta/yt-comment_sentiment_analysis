import json
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from scipy import sparse
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

all_params = yaml.safe_load(open(ROOT / "params.yaml"))


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def main() -> None:
    model = pickle.load(open(MODEL_DIR / "stacking_model.pkl", "rb"))
    X_test = sparse.load_npz(DATA_DIR / "X_test.npz")
    y_test = np.load(DATA_DIR / "y_test.npy")

    pred = model.predict(X_test)
    f1_each = f1_score(y_test, pred, average=None, labels=[-1, 0, 1])
    metrics = {
        "macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1_negative": round(float(f1_each[0]), 4),
        "f1_neutral": round(float(f1_each[1]), 4),
        "f1_positive": round(float(f1_each[2]), 4),
    }
    json.dump(metrics, open(ROOT / "metrics.json", "w"), indent=2)

    REPORT_DIR.mkdir(exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=[-1, 0, 1], cmap="Blues")
    plt.title("Test confusion matrix")
    cm_path = REPORT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")

    # ---- MLflow ----
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment(all_params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="stacking"):
        mlflow.log_params(flatten({k: v for k, v in all_params.items() if k != "mlflow"}))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(MODEL_DIR / "tfidf_vectorizer.pkl"))
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

    print(metrics)


if __name__ == "__main__":
    main()