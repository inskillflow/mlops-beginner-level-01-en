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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("The set tracking URI is", mlflow.get_tracking_uri())

    exp = mlflow.set_experiment(experiment_name="experiment_4")
    print("Name              :", exp.name)
    print("Experiment_id     :", exp.experiment_id)
    print("Artifact Location :", exp.artifact_location)
    print("Tags              :", exp.tags)
    print("Lifecycle_stage   :", exp.lifecycle_stage)
    print("Creation timestamp:", exp.creation_time)

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    os.makedirs("data", exist_ok=True)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha, l1_ratio = args.alpha, args.l1_ratio

    mlflow.start_run()

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    preds = lr.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, preds)

    print("Elasticnet (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE:  %s" % mae)
    print("  R2:   %s" % r2)

    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

    mlflow.sklearn.log_model(lr, "my_new_model_1")

    mlflow.log_artifacts("data/")

    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)

    mlflow.end_run()

    run = mlflow.last_active_run()
    print("Active run id   :", run.info.run_id)
    print("Active run name :", run.info.run_name)
