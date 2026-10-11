import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "yt-sentiment-stacking"
MODEL_URI = f"models:/{MODEL_NAME}@staging"     # the candidate we are testing

MIN_MACRO_F1 = 0.80      # overall quality bar
MIN_CLASS_F1 = 0.70      # no single class may be terrible

mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
)


@pytest.fixture(scope="module")
def model():
    # downloaded once, reused by all tests in this file
    return mlflow.sklearn.load_model(MODEL_URI)


@pytest.fixture(scope="module")
def test_data():
    X = sparse.load_npz(ROOT / "data" / "processed" / "X_test.npz")
    y = np.load(ROOT / "data" / "processed" / "y_test.npy")
    return X, y


def test_model_loads(model):
    """1) The staging model can be loaded from the registry and can predict."""
    assert model is not None
    assert hasattr(model, "predict")


def test_model_signature(model, test_data):
    """2) The model has a signature, and it matches our real data."""
    X, _ = test_data
    signature = mlflow.models.get_model_info(MODEL_URI).signature
    assert signature is not None, "model was logged without a signature"

    expected_shape = tuple(signature.inputs.inputs[0].shape)
    assert expected_shape == (-1, X.shape[1])

    # a correctly shaped input must give one valid label per row
    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert set(np.unique(preds)).issubset({-1, 0, 1})


def test_model_performance(model, test_data):
    """3) The model is good enough on the held-out test set."""
    X, y = test_data
    pred = model.predict(X)

    macro_f1 = f1_score(y, pred, average="macro")
    assert macro_f1 >= MIN_MACRO_F1, f"macro F1 {macro_f1:.3f} < {MIN_MACRO_F1}"

    per_class = f1_score(y, pred, average=None, labels=[-1, 0, 1])
    for label, score in zip([-1, 0, 1], per_class):
        assert score >= MIN_CLASS_F1, f"F1 for class {label} is only {score:.3f}"

    pytestmark = pytest.mark.skipif(
    "MLFLOW_TRACKING_URI" not in os.environ,
    reason="needs the MLflow server (set MLFLOW_TRACKING_URI)",
)