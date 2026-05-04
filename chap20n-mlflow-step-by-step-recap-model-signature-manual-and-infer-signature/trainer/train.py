import argparse
import logging
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha",    type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("The set tracking URI is", mlflow.get_tracking_uri())

    exp = mlflow.set_experiment(experiment_name="experiment_signature")
    print(f"Name              : {exp.name}")
    print(f"Experiment_id     : {exp.experiment_id}")
    print(f"Artifact Location : {exp.artifact_location}")
    print(f"Tags              : {exp.tags}")
    print(f"Lifecycle_stage   : {exp.lifecycle_stage}")
    print(f"Creation timestamp: {exp.creation_time}")

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    os.makedirs("data", exist_ok=True)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    train_x = train.drop(["quality"], axis=1)
    test_x  = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y  = test[["quality"]]

    alpha, l1_ratio = args.alpha, args.l1_ratio

    mlflow.start_run()

    mlflow.set_tags({
        "engineering":       "ML platform",
        "release.candidate": "RC1",
        "release.version":   "2.0",
    })

    mlflow.sklearn.autolog(
        log_input_examples=False,
        log_model_signatures=False,
        log_models=False,
    )

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)

    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)
    print(f"  Elasticnet (alpha={alpha}, l1_ratio={l1_ratio})")
    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    mlflow.log_metrics({"test_rmse": rmse, "test_r2": r2, "test_mae": mae})

    signature = infer_signature(test_x, predicted_qualities)

    input_example = test_x.head(5)

    mlflow.sklearn.log_model(
        lr,
        "elasticnet_model",
        signature=signature,
        input_example=input_example,
    )

    mlflow.log_artifact("data/red-wine-quality.csv")

    print("Artifact path:", mlflow.get_artifact_uri())

    mlflow.end_run()

    run = mlflow.last_active_run()
    print(f"Active run id   : {run.info.run_id}")
    print(f"Active run name : {run.info.run_name}")
