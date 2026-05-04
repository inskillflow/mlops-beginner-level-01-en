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


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha",    type=float, required=False, default=0.4)
    parser.add_argument("--l1_ratio", type=float, required=False, default=0.4)
    return parser.parse_args()


def main():
    args = parse_args()

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    train_x = train.drop(["quality"], axis=1)
    test_x  = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y  = test[["quality"]]

    alpha, l1_ratio = args.alpha, args.l1_ratio

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("The set tracking URI is", mlflow.get_tracking_uri())

    experiment = mlflow.set_experiment(experiment_name="Project exp 1")
    print("Name          :", experiment.name)
    print("Experiment_id :", experiment.experiment_id)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        inner = mlflow.last_active_run()
        print("Active run_id (run)   :", run.info.run_id)
        print("Active run_id (inner) :", inner.info.run_id)
        print("Active run name       :", run.info.run_name)

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        print(f"ElasticNet model (alpha={alpha:.4f}, l1_ratio={l1_ratio:.4f})")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  MAE  = {mae:.4f}")
        print(f"  R2   = {r2:.4f}")

        mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
        mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})

        mlflow.sklearn.log_model(lr, artifact_path="model")

    finished = mlflow.last_active_run()
    print("After the block:")
    print("  run_id  :", finished.info.run_id)
    print("  status  :", finished.info.status)
    print("  end_time:", finished.info.end_time)


if __name__ == "__main__":
    main()
