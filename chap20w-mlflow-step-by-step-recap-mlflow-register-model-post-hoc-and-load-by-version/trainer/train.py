import argparse
import logging
import os
import warnings

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha",    type=float, required=False, default=0.6)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.6)
args = parser.parse_args()

REGISTERED_NAME = "elastic-api-2"


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

    exp = mlflow.set_experiment(experiment_name="experiment_register_model_api")
    print(
        f"Experiment details - Name: {exp.name}, "
        f"ID: {exp.experiment_id}, "
        f"Artifact Location: {exp.artifact_location}"
    )

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
        f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}): "
        f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}"
    )

    mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})

    model_info = mlflow.sklearn.log_model(lr, artifact_path="model")
    print("Model URI (logged):", model_info.model_uri)

    run = mlflow.active_run()
    mv = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        name=REGISTERED_NAME,
    )
    print(f"Registered: name={mv.name!r}  version={mv.version!r}  status={mv.status}")

    registry_uri = f"models:/{mv.name}/{mv.version}"
    print("Loading from registry:", registry_uri)
    loaded = mlflow.pyfunc.load_model(model_uri=registry_uri)

    loaded_predictions = loaded.predict(test_x)
    l_rmse, l_mae, l_r2 = eval_metrics(test_y, loaded_predictions)
    print(
        f"Registered Model Evaluation - "
        f"RMSE_test={l_rmse:.4f}  MAE_test={l_mae:.4f}  R2_test={l_r2:.4f}"
    )

    mlflow.log_metrics({
        "registry_test_rmse": l_rmse,
        "registry_test_mae":  l_mae,
        "registry_test_r2":   l_r2,
    })

    assert np.allclose(predicted_qualities, loaded_predictions), \
        "Registered model produces different predictions!"
    print("Sanity check OK: in-process and registry-loaded predictions match.")

    mlflow.end_run()
    run = mlflow.last_active_run()
    print(f"Active run id   : {run.info.run_id}")
