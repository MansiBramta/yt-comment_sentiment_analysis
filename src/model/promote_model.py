import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "yt-sentiment-stacking"


def main() -> None:
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    )
    client = MlflowClient()

    # the candidate that passed the tests
    staging = client.get_model_version_by_alias(MODEL_NAME, "staging")

    # the model currently serving (may not exist the very first time)
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        current = None

    if current is not None and current.version == staging.version:
        print(f"Version {staging.version} is already champion. Nothing to do.")
        return

    # keep the old champion for rollback
    if current is not None:
        client.set_registered_model_alias(MODEL_NAME, "previous", current.version)

    client.set_registered_model_alias(MODEL_NAME, "champion", staging.version)
    print(
        f"Promoted version {staging.version} to champion "
        f"(previous champion: {current.version if current else 'none'})"
    )


if __name__ == "__main__":
    main()
    