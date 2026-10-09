import pickle
from pathlib import Path

import numpy as np
import yaml
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

p = yaml.safe_load(open(ROOT / "params.yaml"))["model_building"]


def build_model() -> StackingClassifier:
    rs = p["random_state"]
    base_models = [
        ("lgbm", LGBMClassifier(**p["lightgbm"], class_weight="balanced",
                                random_state=rs, n_jobs=-1, verbose=-1)),
        ("svc", CalibratedClassifierCV(
            LinearSVC(C=p["linear_svc_C"], class_weight="balanced", max_iter=5000), cv=3)),
        ("lr", LogisticRegression(C=p["logistic_regression_C"],
                                  class_weight="balanced", max_iter=2000)),
        ("nb", ComplementNB(alpha=p["complement_nb_alpha"])),
    ]
    return StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=2000),
        stack_method="predict_proba",
        cv=p["cv_folds"],
    )


def main() -> None:
    X_train = sparse.load_npz(DATA_DIR / "X_train.npz")
    y_train = np.load(DATA_DIR / "y_train.npy")

    model = build_model()
    print("training stacking model... (takes a while)")
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "stacking_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("saved models/stacking_model.pkl")


if __name__ == "__main__":
    main()