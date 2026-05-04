import argparse
import logging
import os
import pickle
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
parser.add_argument("--alpha",    type=float, required=False, default=0.4)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.4)
args = parser.parse_args()


def get_path_type(path):
    """Classify a filesystem path. Useful at the top of a training script
    when you want to verify what was mounted into the container."""
    if os.path.isabs(path) and os.path.exists(path):
        if os.path.isdir(path):
            return "directory"
        return "file"
    return "not a valid path"


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

    data_csv = os.path.abspath("data/red-wine-quality.csv")
    print(f"data/red-wine-quality.csv -> {get_path_type(data_csv)}")
    print(f"/code -> {get_path_type('/code')}")

    exp = mlflow.set_experiment(experiment_name="experiment_elastic_net_mlflow")
    print(f"Name              : {exp.name}")
    print(f"Experiment_id     : {exp.experiment_id}")
    print(f"Artifact Location : {exp.artifact_location}")

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data, test_size=0.25)

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

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    predicted_qualities = lr.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

    print(
        f"ElasticNet (alpha={alpha}, l1_ratio={l1_ratio}): "
        f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}"
    )

    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

    model_info = mlflow.sklearn.log_model(lr, artifact_path="model")
    print("Model URI (canonical) :", model_info.model_uri)

    pickle_filename = "elastic-net-regression.pkl"
    with open(pickle_filename, "wb") as f:
        pickle.dump(lr, f)
    mlflow.log_artifact(pickle_filename)
    print(f"Side-car pickle logged: {pickle_filename}")

    mlflow.end_run()

    run = mlflow.last_active_run()
    print(f"Active run id     : {run.info.run_id}")
