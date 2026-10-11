import json
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "yt-sentiment-stacking"


def main() -> None:
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    )
    info = json.load(open(ROOT / "reports" / "experiment_info.json"))

    # 1) add the model to the registry as a new version
    version = mlflow.register_model(model_uri=info["model_uri"], name=MODEL_NAME)

    client = MlflowClient()
    # 2) remember which run produced it
    client.set_model_version_tag(MODEL_NAME, version.version, "run_id", info["run_id"])
    # 3) mark it as a candidate (NOT champion yet)
    client.set_registered_model_alias(MODEL_NAME, "staging", version.version)

    print(f"Registered {MODEL_NAME} version {version.version} with alias 'staging'")


if __name__ == "__main__":
    main()