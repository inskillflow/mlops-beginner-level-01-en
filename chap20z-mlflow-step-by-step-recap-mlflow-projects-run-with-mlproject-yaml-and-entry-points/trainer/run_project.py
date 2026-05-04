"""Launcher: builds and runs the MLproject in this directory."""

import os
import mlflow

PARAMETERS = {
    "alpha":    0.3,
    "l1_ratio": 0.3,
}
EXPERIMENT_NAME = "Project exp 1"
ENTRY_POINT     = "ElasticNet"


if __name__ == "__main__":
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("Tracking URI :", mlflow.get_tracking_uri())
    print("Project URI  : .  (current directory)")
    print("Entry point  :", ENTRY_POINT)
    print("Parameters   :", PARAMETERS)

    submitted = mlflow.projects.run(
        uri=".",
        entry_point=ENTRY_POINT,
        parameters=PARAMETERS,
        experiment_name=EXPERIMENT_NAME,
        env_manager="local",
        synchronous=True,
    )

    print("\nProject finished.")
    print("Run id     :", submitted.run_id)
    print("Run status :", submitted.get_status())
