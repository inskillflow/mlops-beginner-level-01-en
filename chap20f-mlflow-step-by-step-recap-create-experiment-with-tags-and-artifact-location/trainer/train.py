import argparse
import logging
import os
import warnings
from pathlib import Path

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
parser.add_argument("--alpha", type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.5)
parser.add_argument("--exp-name", type=str, required=False,
                    default="exp_create_exp_artifact")
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
    print("Tracking URI:", mlflow.get_tracking_uri())

    artifact_root = Path("/mlflow/myartifacts")
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_uri = artifact_root.as_uri()
    print("Artifact location URI:", artifact_uri)

    try:
        exp_id = mlflow.create_experiment(
            name=args.exp_name,
            tags={"version": "v1", "priority": "p1"},
            artifact_location=artifact_uri,
        )
        print(f"Created experiment '{args.exp_name}' with id={exp_id}")
    except mlflow.exceptions.MlflowException as e:
        exp = mlflow.set_experiment(args.exp_name)
        exp_id = exp.experiment_id
        print(f"Experiment already exists: id={exp_id} ({e.error_code})")

    get_exp = mlflow.get_experiment(exp_id)
    print("Name              :", get_exp.name)
    print("Experiment_id     :", get_exp.experiment_id)
    print("Artifact Location :", get_exp.artifact_location)
    print("Tags              :", get_exp.tags)
    print("Lifecycle_stage   :", get_exp.lifecycle_stage)
    print("Creation timestamp:", get_exp.creation_time)

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha, l1_ratio = args.alpha, args.l1_ratio

    with mlflow.start_run(experiment_id=exp_id):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        preds = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, preds)

        print("Elasticnet (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE:  %s" % mae)
        print("  R2:   %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(lr, "model")
