"""Loads a model trained outside MLflow and registers it in the MLflow registry."""

import os
import pickle

import mlflow
import mlflow.sklearn

REGISTERED_NAME = "elastic-net-regression-outside-mlflow"
PICKLE_PATH = "/shared/elastic-net-regression.pkl"


if __name__ == "__main__":
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("[registrar] Tracking URI:", mlflow.get_tracking_uri())

    if not os.path.exists(PICKLE_PATH):
        raise SystemExit(
            f"[registrar] {PICKLE_PATH} not found. "
            f"Run `docker compose run --rm pretrainer` first."
        )

    with open(PICKLE_PATH, "rb") as f:
        loaded_model = pickle.load(f)
    print(f"[registrar] Loaded model from {PICKLE_PATH}: {type(loaded_model).__name__}")

    exp = mlflow.set_experiment(experiment_name="experiment_register_outside")
    print(f"[registrar] Experiment: {exp.name} (id={exp.experiment_id})")

    with mlflow.start_run() as run:
        mlflow.set_tags({
            "imported": "true",
            "source":   "external_pickle",
            "filename": os.path.basename(PICKLE_PATH),
        })

        model_info = mlflow.sklearn.log_model(
            sk_model=loaded_model,
            artifact_path="model",
            serialization_format="cloudpickle",
            registered_model_name=REGISTERED_NAME,
        )

        print(f"[registrar] Logged model URI: {model_info.model_uri}")
        print(f"[registrar] Registered as     : {REGISTERED_NAME!r}")
        print(f"[registrar] Run id            : {run.info.run_id}")
