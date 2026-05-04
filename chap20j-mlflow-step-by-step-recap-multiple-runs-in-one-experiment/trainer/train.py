import argparse
import logging
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


COMMON_TAGS = {
    "engineering":       "ML platform",
    "release.candidate": "RC1",
    "release.version":   "2.0",
}


def train_one_run(run_name, alpha, l1_ratio, train_x, train_y, test_x, test_y):
    """Train ONE ElasticNet model and log everything to MLflow under `run_name`."""
    mlflow.start_run(run_name=run_name)

    mlflow.set_tags(COMMON_TAGS)

    current = mlflow.active_run()
    print(f"\n>>> Run started: name={current.info.run_name}, id={current.info.run_id}")

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    preds = lr.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, preds)

    print(f"  ElasticNet(alpha={alpha:.2f}, l1_ratio={l1_ratio:.2f})  "
          f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
    mlflow.sklearn.log_model(lr, "my_new_model_1")
    mlflow.log_artifacts("data/")

    print(f"  Artifact path: {mlflow.get_artifact_uri()}")

    mlflow.end_run()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("The set tracking URI is", mlflow.get_tracking_uri())

    exp = mlflow.set_experiment(experiment_name="experiment_5")
    print(f"Name: {exp.name}")
    print(f"Experiment_id: {exp.experiment_id}")

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    os.makedirs("data", exist_ok=True)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    CONFIGS = [
        ("run1.1", args.alpha, args.l1_ratio),
        ("run2.1", 0.9,        0.9),
        ("run3.1", 0.4,        0.4),
    ]

    for name, alpha, l1 in CONFIGS:
        train_one_run(name, alpha, l1, train_x, train_y, test_x, test_y)

    run = mlflow.last_active_run()
    print(f"\nRecent active run id   : {run.info.run_id}")
    print(f"Recent active run name : {run.info.run_name}")
